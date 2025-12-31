import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
import math
from typing import List, Optional, Sequence, Union, Any, Dict
from dataclasses import dataclass
from transformers import MimiModel, AutoFeatureExtractor, HubertModel, AutoModel, Wav2Vec2FeatureExtractor
import os
import json
from huggingface_hub import snapshot_download

try:
    from semantic_module import Encoder, Decoder
except:
    from .semantic_module import Encoder, Decoder


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    if in_channels % num_groups != 0:
        for divisor in range(min(num_groups, in_channels), 0, -1):
            if in_channels % divisor == 0:
                num_groups = divisor
                break
        if in_channels % num_groups != 0:
            num_groups = 1
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Snake1d(nn.Module):
    """Snake activation from the DAC paper."""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
    
    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-9)) * torch.sin(self.alpha * x).pow(2)


def WNConv1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResidualUnit(nn.Module):
    """Multi-receptive field residual unit with dilated convolutions."""
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class DACDecoderBlock(nn.Module):
    """DAC-style decoder block with Snake activations and multi-scale residual units."""
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class DACStyleDecoder(nn.Module):
    """DAC-style decoder for upsampling from latent codes to high sample rate audio."""
    def __init__(
        self,
        input_channels: int,
        decoder_dim: int,
        upsample_rates: List[int],
        d_out: int = 1,
    ):
        super().__init__()
        
        layers = [WNConv1d(input_channels, decoder_dim, kernel_size=7, padding=3)]
        
        for i, stride in enumerate(upsample_rates):
            input_dim = decoder_dim // (2 ** i)
            output_dim = decoder_dim // (2 ** (i + 1))
            layers.append(DACDecoderBlock(input_dim, output_dim, stride))
        
        final_dim = decoder_dim // (2 ** len(upsample_rates))
        layers += [
            Snake1d(final_dim),
            WNConv1d(final_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(m):
    """Applies truncated normal initialization to Conv and Linear layers."""
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.trunc_normal_(m.weight, std=0.02)


class MimiUpsamplerWrapper(nn.Module):
    def __init__(
        self,
        model_name: str = "kyutai/mimi",
        output_sample_rate: int = 44100,
        encoder_sample_rate: int = 24000,
        frame_rate: float = 12.5,  # Use Mimi's actual frame rate
        decoder_dim: int = 1024,
        upsample_ratio: Optional[List[int]] = None,
        target_bandwidths: Sequence[Union[int, float]] = [1.5],
        semantic_teacher: str = "hubert_base_general",
        last_layer_semantic: bool = True,
        downsample_mode: str = "step_down",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        
        self.device = device
        self._quantizer_frozen = True
        self.semantic_teacher = semantic_teacher
        self.last_layer_semantic = last_layer_semantic
        self.downsample_mode = downsample_mode
        
        # ==================== Load Mimi ====================
        self.mimi = MimiModel.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Use Mimi's config values directly
        self.mimi_config = self.mimi.config
        self.encoder_sample_rate = self.mimi_config.sampling_rate  # 24000
        self.output_sample_rate = output_sample_rate
        self.target_bandwidths = target_bandwidths
        
        # Get dimensions from Mimi config
        self.latent_dim = self.mimi_config.hidden_size  # 512
        self.n_q = getattr(self.mimi_config, 'num_quantizers', 8)
        
        # Use frame_rate from Mimi config (12.5 Hz)
        self._frame_rate = self.mimi_config.frame_rate  # 12.5
        self.hop_length = int(self.encoder_sample_rate / self._frame_rate)  # 1920
        
        print(f"Mimi config:")
        print(f"  - Encoder sample rate: {self.encoder_sample_rate}")
        print(f"  - Frame rate: {self._frame_rate} Hz")
        print(f"  - Latent dim (hidden_size): {self.latent_dim}")
        print(f"  - Num quantizers: {self.n_q}")
        print(f"  - Codebook size: {self.mimi_config.codebook_size}")
        
        # ==================== Setup Semantic Model ====================
        self._setup_semantic_model()
        
        # Semantic model runs at 50 Hz (16kHz / 320)
        # Mimi runs at 12.5 Hz
        # Downsample factor: 50 / 12.5 = 4
        semantic_frame_rate = self.semantic_sample_rate / 320  # 50 Hz for HuBERT
        self.semantic_downsample_factor = int(round(semantic_frame_rate / self._frame_rate))
        print(f"Semantic downsample factor: {self.semantic_downsample_factor}")
        
        # ==================== Semantic Encoder/Decoder ====================
        self.encoder_semantic = Encoder(
            input_channels=self.semantic_dim,
            encode_channels=self.encoder_semantic_dim
        )
        self.decoder_semantic = Decoder(
            code_dim=self.encoder_semantic_dim,
            output_channels=self.semantic_dim,
            decode_channels=self.semantic_dim
        )
        
        # Project from Mimi latent space to semantic space
        self.fc_post1 = nn.Linear(self.latent_dim, self.encoder_semantic_dim)
        
        # ==================== Calculate Upsampling ====================
        self.is_upsampling_model = True
        total_upsample_factor = int(self.output_sample_rate / self._frame_rate)
        
        print(f"\nUpsampling calculation:")
        print(f"  - Output sample rate: {self.output_sample_rate}")
        print(f"  - Frame rate: {self._frame_rate}")
        print(f"  - Required upsample factor: {total_upsample_factor}x")
        
        if upsample_ratio is None:
            upsample_ratio = self._factorize_upsample(total_upsample_factor)
            print(f"  - Auto-calculated upsample_ratio: {upsample_ratio}")
        
        actual_factor = int(np.prod(upsample_ratio))
        if actual_factor != total_upsample_factor:
            raise ValueError(
                f"Product of upsample_ratio {upsample_ratio} = {actual_factor}, "
                f"but need {total_upsample_factor}x for {self._frame_rate}Hz -> {self.output_sample_rate}Hz"
            )
        
        print(f"  - Using upsample_ratio: {upsample_ratio} (product: {actual_factor})")
        
        # ==================== DAC-style Decoder ====================
        self.dac_decoder = DACStyleDecoder(
            input_channels=self.latent_dim,  # 512 for Mimi
            decoder_dim=decoder_dim,
            upsample_rates=upsample_ratio,
            d_out=1,
        )
        
        # Initialize trainable weights
        self.dac_decoder.apply(init_weights)
        self.encoder_semantic.apply(init_weights)
        self.decoder_semantic.apply(init_weights)
        self.fc_post1.apply(init_weights)
        
        self._report_params()
    
    def _report_params(self):
        """Report parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\nParameter counts (before freezing):")
        print(f"  - Total: {total / 1e6:.2f}M")
        print(f"  - DAC decoder: {sum(p.numel() for p in self.dac_decoder.parameters()) / 1e6:.2f}M")
        print(f"  - Semantic encoder: {sum(p.numel() for p in self.encoder_semantic.parameters()) / 1e6:.2f}M")
        print(f"  - Semantic decoder: {sum(p.numel() for p in self.decoder_semantic.parameters()) / 1e6:.2f}M")
        print(f"  - fc_post1: {sum(p.numel() for p in self.fc_post1.parameters()) / 1e6:.2f}M")
    
    def _setup_semantic_model(self):
        """Initialize the semantic teacher model."""
        if self.semantic_teacher == "hubert_base_general":
            self.semantic_model = HubertModel.from_pretrained(
                "facebook/hubert-base-ls960"  # or "bosonai/hubert_base" if available
            )
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768
            self.semantic_fe = None
            
        elif self.semantic_teacher == "hubert_base":
            self.semantic_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768
            self.semantic_fe = None
            
        elif self.semantic_teacher == "mert_music":
            self.semantic_model = AutoModel.from_pretrained(
                "m-a-p/MERT-v1-95M", trust_remote_code=True
            )
            self.semantic_fe = Wav2Vec2FeatureExtractor.from_pretrained(
                "m-a-p/MERT-v1-95M", trust_remote_code=True
            )
            self.semantic_sample_rate = self.semantic_fe.sampling_rate  # 24000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768
            
        else:
            raise NotImplementedError(
                f"Semantic teacher '{self.semantic_teacher}' not implemented. "
                f"Supported: hubert_base_general, hubert_base, mert_music"
            )
        
        # Freeze semantic model
        self.semantic_model.eval()
        for param in self.semantic_model.parameters():
            param.requires_grad = False
        
        print(f"Loaded semantic teacher: {self.semantic_teacher}")
        print(f"Semantic sample rate: {self.semantic_sample_rate}")
        print(f"Semantic dim: {self.semantic_dim}")
    
    def _factorize_upsample(self, n: int) -> List[int]:
        """Factorize upsampling ratio into reasonable strides."""
        factors = []
        remaining = n
        
        for f in [7, 5, 4, 3, 2]:
            while remaining % f == 0 and remaining > 1:
                factors.append(f)
                remaining //= f
        
        if remaining > 1:
            factors.append(remaining)
        
        factors.sort(reverse=True)
        return factors
    
    def freeze_for_upsampling_finetune(self):
        """Freeze Mimi encoder/quantizer, keep decoder and semantic modules trainable."""
        print("Freezing Mimi encoder, decoder, and quantizer for upsampling fine-tune...")
        
        # Freeze all of Mimi
        for param in self.mimi.parameters():
            param.requires_grad = False
        self.mimi.eval()
        
        # Semantic model is already frozen in _setup_semantic_model
        
        # Trainable modules
        trainable_modules = [
            self.dac_decoder,
            self.encoder_semantic,
            self.decoder_semantic,
            self.fc_post1,
        ]
        
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        
        # Report
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        print("-----------------------------")
        print(f"Total parameters: {total / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable / 1e6:.2f}M")
        print(f"Frozen parameters: {frozen / 1e6:.2f}M")
        print("-----------------------------\n")
    
    def train(self, mode: bool = True):
        """Override to keep frozen components in eval mode."""
        super().train(mode)
        # Keep Mimi in eval mode
        self.mimi.eval()
        # Keep semantic model in eval mode
        self.semantic_model.eval()
        return self
    
    @property
    def frame_rate(self) -> float:
        return self._frame_rate
    
    @property
    def tps(self) -> float:
        return self._frame_rate
    
    @property
    def sampling_rate(self) -> int:
        return self.output_sample_rate
    
    @property
    def num_codebooks(self) -> int:
        return self.n_q
    
    @property
    def codebook_size(self) -> int:
        return self.mimi_config.codebook_size
    
    @torch.no_grad()
    def get_regress_target(self, x: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Extract semantic features from the frozen semantic model.
        
        Args:
            x: Input audio tensor (B, 1, T) or (B, T)
            orig_sr: Original sample rate of x
            
        Returns:
            Semantic features (B, T_semantic, semantic_dim)
        """
        # Resample to semantic model's sample rate
        if x.dim() == 3:
            x = x[:, 0, :]  # (B, T)
        
        x = torchaudio.functional.resample(x, orig_sr, self.semantic_sample_rate)
        
        if self.semantic_teacher in ["hubert_base", "hubert_base_general", "hubert_base_kushinada", 
                                      "wavlm_base_plus", "mHubert_base"]:
            # Pad for HuBERT-style models
            x = F.pad(x, (160, 160))
            target = self.semantic_model(x, output_hidden_states=True).hidden_states
            target = torch.stack(target, dim=1)  # (B, num_layers, T, dim)
            # Average across layers
            target = target.mean(1)  # (B, T, dim)
            
        elif self.semantic_teacher == "mert_music":
            # Process with MERT feature extractor
            x_np = x.float().cpu().numpy()
            inputs = self.semantic_fe(
                x_np,
                sampling_rate=self.semantic_fe.sampling_rate,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs["input_values"].to(x.device)
            
            outputs = self.semantic_model(input_values=input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            if self.last_layer_semantic:
                target = hidden_states[-1]
            else:
                target = torch.stack(hidden_states, dim=1).mean(1)
        else:
            raise NotImplementedError(f"Semantic teacher {self.semantic_teacher} not implemented")
        
        # Downsample to match Mimi's frame rate
        if self.downsample_mode == "step_down":
            if self.semantic_downsample_factor > 1:
                target = target[:, ::self.semantic_downsample_factor, :]
        elif self.downsample_mode == "avg":
            # Average pooling
            target = target.transpose(1, 2)  # (B, dim, T)
            target = F.avg_pool1d(
                target, 
                kernel_size=self.semantic_downsample_factor,
                stride=self.semantic_downsample_factor
            )
            target = target.transpose(1, 2)  # (B, T, dim)
        
        return target
    
    def _get_mimi_quantized(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get quantized representations from Mimi encoder.
        
        Args:
            x: Input audio (B, 1, T) at encoder_sample_rate (24kHz)
            
        Returns:
            Quantized features (B, hidden_size, seq_len) where hidden_size=512
        """
        with torch.no_grad():
            # Use Mimi's encode method to get codes
            encoder_outputs = self.mimi.encode(x)
            codes = encoder_outputs.audio_codes  # (B, num_quantizers, seq_len)
            
            # Decode codes to get continuous quantized representation
            quantized = self.mimi.quantizer.decode(codes)  # (B, hidden_size, seq_len)
            
            return quantized


    def forward(self, x: torch.Tensor, bw: float = None) -> tuple:
        """
        Forward pass for training.
        
        Args:
            x: Input audio tensor (B, 1, T) at output_sample_rate
            bw: Bandwidth (unused, for compatibility)
        
        Returns:
            output: Reconstructed audio (B, 1, T)
            commit_loss: Commitment loss (0 since Mimi quantizer is frozen)
            semantic_loss: Semantic reconstruction loss
            aux: Auxiliary outputs (None)
        """
        target_length = x.shape[-1]
        
        # ==================== Get Semantic Target ====================
        e_semantic_input = self.get_regress_target(x, orig_sr=self.output_sample_rate).detach()
        # e_semantic_input: (B, T_semantic, semantic_dim)
        
        # Encode semantic features
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        # e_semantic: (B, encoder_semantic_dim, T_semantic)
        
        # ==================== Get Mimi Quantized Features ====================
        # Resample to Mimi's expected sample rate
        if x.dim() == 3:
            x_mono = x[:, 0, :]  # (B, T)
        else:
            x_mono = x
        
        x_resampled = torchaudio.functional.resample(
            x_mono, self.output_sample_rate, self.encoder_sample_rate
        )
        x_input = x_resampled.unsqueeze(1)  # (B, 1, T)
        
        # Get quantized features from Mimi (frozen)
        quantized = self._get_mimi_quantized(x_input)
        # quantized: (B, hidden_size, seq_len) where hidden_size=512
        
        # ==================== Align Lengths ====================
        min_len = min(quantized.shape[2], e_semantic.shape[2])
        quantized = quantized[:, :, :min_len]
        e_semantic = e_semantic[:, :, :min_len]
        
        # ==================== Semantic Branch ====================
        # Project quantized to semantic space
        quantized_for_semantic = self.fc_post1(quantized.transpose(1, 2)).transpose(1, 2)
        # quantized_for_semantic: (B, encoder_semantic_dim, min_len)
        
        # Decode semantic features
        o_semantic = self.decoder_semantic(quantized_for_semantic)
        # o_semantic: (B, semantic_dim, min_len)
        
        # Compute semantic reconstruction loss
        semantic_target = e_semantic_input[:, :min_len, :].transpose(1, 2).detach()
        semantic_recon_loss = F.mse_loss(semantic_target, o_semantic)
        
        # ==================== Audio Decoding ====================
        # Use quantized features directly for audio decoding
        output = self.dac_decoder(quantized)
        # output: (B, 1, T_output)
        
        # Adjust output length to match target
        if output.shape[-1] != target_length:
            if output.shape[-1] > target_length:
                output = output[..., :target_length]
            else:
                output = F.pad(output, (0, target_length - output.shape[-1]))
        
        # Commitment loss is 0 since Mimi quantizer is frozen
        commit_loss = torch.tensor(0.0, device=x.device)
        
        return output, commit_loss, semantic_recon_loss, None

    
    @torch.no_grad()
    def encode(self, audio_path_or_wv, sr=None, loudness_normalize=False, loudness_threshold=-23.0):
        """Encode audio to discrete codes using Mimi."""
        if isinstance(audio_path_or_wv, str):
            wv, sr = librosa.load(audio_path_or_wv, mono=True, sr=None)
        else:
            wv = audio_path_or_wv
            assert sr is not None
        
        if loudness_normalize:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(wv)
            wv = pyln.normalize.loudness(wv, loudness, loudness_threshold)
        
        # Resample to Mimi's sample rate
        if sr != self.encoder_sample_rate:
            wv = librosa.resample(wv, orig_sr=sr, target_sr=self.encoder_sample_rate)
        
        # Prepare tensor
        wv_tensor = torch.from_numpy(wv).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Encode with Mimi
        encoder_outputs = self.mimi.encode(wv_tensor)
        codes = encoder_outputs.audio_codes  # (batch, n_q, seq_len)
        
        return codes[0]  # Remove batch dim
        
    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> np.ndarray:
        """Decode discrete codes to audio using our upsampling decoder."""
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)

        quantized = self.mimi.quantizer.decode(codes)
        
        output = self.dac_decoder(quantized)
        
        return output.cpu().numpy()


def load_mimi_upsampler(
    checkpoint_path: str = None,
    config_path: str = None,
    model_name: str = "kyutai/mimi",
    output_sample_rate: int = 44100,
    decoder_dim: int = 1024,
    upsample_ratio: list = None,
    semantic_teacher: str = "hubert_base_general",
    device: str = "cuda",
) -> MimiUpsamplerWrapper:
    """
    Load MimiUpsamplerWrapper for inference.
    
    Args:
        checkpoint_path: Path to trained checkpoint (optional)
        config_path: Path to config.json (optional, used with checkpoint)
        model_name: HuggingFace model name for Mimi
        output_sample_rate: Target sample rate
        decoder_dim: Decoder dimension
        upsample_ratio: Upsampling ratios
        semantic_teacher: Semantic model to use
        device: Device to load model on
    
    Returns:
        Loaded model ready for inference
    """
    # If config provided, load from it
    if config_path is not None:
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_name = config.get('mimi_model_name', model_name)
        output_sample_rate = config.get('sample_rate', output_sample_rate)
        decoder_dim = config.get('decoder_dim', decoder_dim)
        upsample_ratio = config.get('upsample_ratio', upsample_ratio)
        semantic_teacher = config.get('semantic_teacher', semantic_teacher)
    
    # Instantiate model
    model = MimiUpsamplerWrapper(
        model_name=model_name,
        output_sample_rate=output_sample_rate,
        decoder_dim=decoder_dim,
        upsample_ratio=upsample_ratio,
        semantic_teacher=semantic_teacher,
        device=device,
    )
    
    # Load trained weights if checkpoint provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully")
    
    model.to(device)
    model.eval()
    return model

def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Handles either:
      - full training checkpoint dict {model_state_dict, optimizer..., config...}
      - raw state_dict
    """
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        # sometimes people save under "state_dict"
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        # could already be a plain state dict
        # (heuristic: value tensors)
        if all(hasattr(v, "shape") for v in ckpt.values()):
            return ckpt
    raise ValueError("Could not extract a model state_dict from checkpoint.")


def _strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Normalizes common wrappers:
      - DDP: 'module.'
      - torch.compile: '_orig_mod.'
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        out[k] = v
    return out


def _load_forgiving(model: torch.nn.Module, sd: Dict[str, torch.Tensor], verbose: bool = True):
    """
    Loads only keys that exist AND match shape.
    This avoids giant spam lists and protects you from config mismatches.
    """
    model_sd = model.state_dict()
    loadable = {}
    shape_mismatch = []
    skipped = 0

    for k, v in sd.items():
        if k not in model_sd:
            skipped += 1
            continue
        if tuple(v.shape) != tuple(model_sd[k].shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        loadable[k] = v

    missing, unexpected = model.load_state_dict(loadable, strict=False)

    if verbose:
        print("---- checkpoint load report ----")
        print(f"keys in checkpoint:      {len(sd)}")
        print(f"keys loadable (matched): {len(loadable)}")
        print(f"skipped (no key):        {skipped}")
        print(f"shape mismatches:        {len(shape_mismatch)}")
        print(f"missing after load:      {len(missing)}")
        print(f"unexpected after load:   {len(unexpected)}")
        if shape_mismatch[:5]:
            print("first few shape mismatches:")
            for k, a, b in shape_mismatch[:5]:
                print(f"  {k}: ckpt{a} != model{b}")
        print("-------------------------------")

    return missing, unexpected


def prepare(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cuda",
    compile_after_load: bool = False,
):
    """
    Correct for your training setup:
      - prefers checkpoint['config'] if present
      - strips module/_orig_mod prefixes
      - loads into the *raw* module (then optionally torch.compile after)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Prefer config embedded in checkpoint (most faithful to training)
    cfg = {}
    if isinstance(ckpt, dict) and "config" in ckpt and isinstance(ckpt["config"], dict):
        cfg = ckpt["config"]
        print("Using config from checkpoint['config'].")
    elif config_path is not None:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        print("Using config from config_path.")
    else:
        raise ValueError("No config found in checkpoint and no config_path provided.")

    model = MimiUpsamplerWrapper(
        model_name=cfg.get("mimi_model_name", "kyutai/mimi"),
        output_sample_rate=cfg.get("sample_rate", 44100),
        encoder_sample_rate=cfg.get("encoder_sample_rate", 24000),
        frame_rate=cfg.get("frame_rate", 12.5),
        decoder_dim=cfg.get("decoder_dim", 1024),
        upsample_ratio=cfg.get("upsample_ratio", None),
        target_bandwidths=cfg.get("target_bandwidths", [1.5]),
        semantic_teacher=cfg.get("semantic_teacher", "hubert_base_general"),
        last_layer_semantic=cfg.get("last_layer_semantic", True),
        downsample_mode=cfg.get("downsample_mode", "step_down"),
        device=device,
    ).to(device)

    sd = _extract_state_dict(ckpt)
    sd = _strip_prefixes(sd)

    _load_forgiving(model, sd, verbose=True)

    model.eval()

    if compile_after_load:
        # IMPORTANT: compile AFTER loading to avoid _orig_mod key hell
        model = torch.compile(model, mode="default")
        model.eval()

    return model


@dataclass
class EncodedResult:
    audio_codes: torch.Tensor  # (B, n_q, T_codes)
    quantized: Optional[torch.Tensor] = None  # (B, hidden, T_codes) if requested

@torch.no_grad()
def encode_batch(
    model,
    x_batch: torch.Tensor,
    orig_sr: Optional[int] = None,
    return_quantized: bool = True,
) -> EncodedResult:
    """
    Batch encoder for Mimi codes.
    x_batch: (B, 1, T) or (B, T)
    """
    if orig_sr is None:
        orig_sr = model.output_sample_rate

    if x_batch.dim() == 2:
        x_batch = x_batch.unsqueeze(1)
    assert x_batch.dim() == 3, f"Expected (B,1,T) or (B,T). Got {tuple(x_batch.shape)}"

    x = x_batch[:, 0, :]  # (B, T)

    if orig_sr != model.encoder_sample_rate:
        x = torchaudio.functional.resample(x, orig_sr, model.encoder_sample_rate)

    x = x.unsqueeze(1)  # (B,1,T24)

    enc_out = model.mimi.encode(x)
    codes = enc_out.audio_codes  # (B,n_q,Tc)

    quantized = None
    if return_quantized:
        quantized = model.mimi.quantizer.decode(codes)  # (B,hidden,Tc)

    return EncodedResult(audio_codes=codes, quantized=quantized)


def load_avadec_audio_tokenizer(tokenizer_name_or_path, device="cuda"):
    is_local = os.path.exists(tokenizer_name_or_path)
    if not is_local:
        tokenizer_path = snapshot_download(tokenizer_name_or_path)
    else:
        tokenizer_path = tokenizer_name_or_path
    config_path = os.path.join(tokenizer_path, "config.json")
    checkpoint_path = os.path.join(tokenizer_path, "model.pth")
    config = json.load(open(config_path))
    model = prepare(checkpoint_path, config_path, device)
    model.eval()
    return model