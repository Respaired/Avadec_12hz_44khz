# Based on code from: https://github.com/zhenye234/xcodec
# Licensed under MIT License
# Modifications by BosonAI

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union, Sequence, List
import numpy as np
from transformers import AutoModel
import torchaudio
import json
import librosa
from huggingface_hub import snapshot_download

from vector_quantize_pytorch import ResidualFSQ
from .descriptaudiocodec.dac.model import dac as dac2
from .quantization.vq import ResidualVectorQuantizer
from .semantic_module import Encoder, Decoder

from transformers import HubertModel, AutoFeatureExtractor, AutoModel, Wav2Vec2BertModel

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


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout, temb_channels=0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


# ==================== DAC-STYLE DECODER COMPONENTS ====================

class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
    
    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-9)) * torch.sin(self.alpha * x).pow(2)


def WNConv1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def WNLinear(*args, **kwargs):
    return nn.utils.weight_norm(nn.Linear(*args, **kwargs))


class ResidualUnit(nn.Module):
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


# ==================== STEREO UPMIX MODULE ====================


class ResidualUnit(nn.Module):
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

# 2. Configure the Module
class StereoUpmixModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, num_layers=8):
        super().__init__()
        
        self.input_proj = WNConv1d(feature_dim, hidden_dim, kernel_size=7, padding=3)
        self.input_snake = Snake1d(hidden_dim)

        layers = []
        dilations = [1, 3, 9, 27] * (num_layers // 4) 
        
        for d in dilations:
            layers.append(ResidualUnit(hidden_dim, dilation=d))
            
        self.body = nn.Sequential(*layers)

        self.output_snake = Snake1d(hidden_dim)
        self.output_conv = WNConv1d(hidden_dim, 1, kernel_size=7, padding=3)
        
        self._init_near_zero_output()
        
    def _init_near_zero_output(self):
        with torch.no_grad():
            self.output_conv.weight_g.fill_(0.01)
            if self.output_conv.bias is not None:
                self.output_conv.bias.zero_()

    def forward(self, features, mono, width_scale=1.0):
        x = self.input_proj(features)
        x = self.input_snake(x)
        x = self.body(x)
        x = self.output_snake(x)
        side = self.output_conv(x) * width_scale
        
        left = mono + side
        right = mono - side
        return torch.cat([left, right], dim=1)

# ==================== DAC-STYLE DECODER WITH STEREO SUPPORT ====================

class DACStyleDecoder(nn.Module):
    """
    DAC-style decoder with optional dedicated stereo upmix module.
    Separates upsampling from spatial hallucination.
    """
    def __init__(
        self,
        input_channels: int,
        decoder_dim: int,
        upsample_rates: List[int],
        stereo_output: bool = False,
        stereo_hidden_dim: int = None,
    ):
        super().__init__()
        
        self.stereo_output = stereo_output
        
        # Backbone - upsampling path
        backbone_layers = [WNConv1d(input_channels, decoder_dim, kernel_size=7, padding=3)]
        
        for i, stride in enumerate(upsample_rates):
            input_dim = decoder_dim // (2 ** i)
            output_dim = decoder_dim // (2 ** (i + 1))
            backbone_layers.append(DACDecoderBlock(input_dim, output_dim, stride))
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        self.final_dim = decoder_dim // (2 ** len(upsample_rates))
        
        # Mono head - produces mono waveform
        self.mono_head = nn.Sequential(
            Snake1d(self.final_dim),
            WNConv1d(self.final_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )
        
        # Stereo upmix module - dedicated spatial learning
        if stereo_output:
            self.stereo_module = StereoUpmixModule(
                feature_dim=self.final_dim,
                hidden_dim=128,
                num_layers=8
            )
        else:
            self.stereo_module = None

    def forward(self, x: torch.Tensor, stereo: bool = None, width_scale: float = 1.0):
        """
        Args:
            x: Input latent (B, C, T)
            stereo: Override stereo output (None = use self.stereo_output)
            width_scale: Stereo width control (only used if stereo=True)
        
        Returns:
            audio: (B, 1, T) if mono, (B, 2, T) if stereo
        """
        if stereo is None:
            stereo = self.stereo_output
        
        # Get features from backbone
        features = self.backbone(x)
        
        # Generate mono output
        mono = self.mono_head(features)
        
        # Optionally upmix to stereo
        if stereo and self.stereo_module is not None:
            return self.stereo_module(features, mono, width_scale)
        
        return mono
    
    def forward_with_intermediates(self, x: torch.Tensor, width_scale: float = 1.0):
        """Return mono, side, and stereo for analysis."""
        features = self.backbone(x)
        mono = self.mono_head(features)
        
        if self.stereo_module is not None:
            side = self.stereo_module.get_side(features, width_scale)
            stereo = self.stereo_module(features, mono, width_scale)
            return mono, side, stereo
        
        return mono, None, None


# ==================== END DAC-STYLE COMPONENTS ====================


def init_weights(m):
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


class EncodedResult:
    def __init__(self, audio_codes):
        self.audio_codes = audio_codes


class HiggsAudioFeatureExtractor(nn.Module):
    def __init__(self, sampling_rate=16000):
        super().__init__()
        self.sampling_rate = sampling_rate

    def forward(self, raw_audio, sampling_rate=16000, return_tensors="pt"):
        audio_signal = torch.tensor(raw_audio)
        audio_signal = audio_signal.unsqueeze(0)
        if len(audio_signal.shape) < 3:
            audio_signal = audio_signal.unsqueeze(0)
        return {"input_values": audio_signal}


class Wav2Vec2BertWithFeatureExtractor(nn.Module):
    def __init__(self, model_name_or_path="facebook/w2v-bert-2.0"):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        self.model = Wav2Vec2BertModel.from_pretrained(model_name_or_path)
    
    def forward(self, waveform, output_hidden_states=False):
        if waveform.dim() == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)
        device = waveform.device
        waveform_np = waveform.float().detach().cpu().numpy()
        inputs = self.feature_extractor(
            waveform_np,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True
        )
        input_features = inputs.input_features.to(device)
        outputs = self.model(
            input_features=input_features,
            output_hidden_states=output_hidden_states
        )
        return outputs


class KanadecAudioTokenizer(nn.Module):
    def __init__(
        self,
        n_filters: int = 32,
        D: int = 128,
        target_bandwidths: Sequence[Union[int, float]] = [1, 1.5, 2, 4, 6],
        ratios: Sequence[int] = [8, 5, 4, 2, 3],
        sample_rate: int = 44100,
        encoder_sample_rate: int = 24000,
        output_sample_rate: int = 44100,
        upsample_ratio: Optional[List[int]] = None,
        decoder_dim: int = 1024,
        bins: int = 1024,
        n_q: int = 8,
        codebook_dim: int = None,
        normalize: bool = False,
        causal: bool = False,
        semantic_techer: str = "hubert_base_general",
        last_layer_semantic: bool = True,
        merge_mode: str = "concat",
        downsample_mode: str = "step_down",
        semantic_mode: str = "classic",
        vq_scale: int = 1,
        semantic_sample_rate: int = None,
        stereo_output: bool = False,
        stereo_hidden_dim: int = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.encoder_sample_rate = encoder_sample_rate
        self.hop_length = np.prod(ratios)
        self.frame_rate = math.ceil(self.encoder_sample_rate / self.hop_length)
        
        self.output_sample_rate = sample_rate
        self.semantic_techer = semantic_techer
        self.stereo_output = stereo_output

        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.encoder = dac2.Encoder(64, ratios, D)

        self.is_upsampling_model = upsample_ratio is not None and len(upsample_ratio) > 0
        self._quantizer_frozen = False
        self._backbone_frozen = False
        
        if not self.is_upsampling_model:
            self.decoder_2 = dac2.Decoder(D, 1024, ratios)
        else:
            self.decoder_2 = None
            
            total_upsample_factor_needed = self.output_sample_rate // self.frame_rate
            if self.output_sample_rate % self.frame_rate != 0:
                raise ValueError(
                    f"Output sample rate ({self.output_sample_rate}) must be divisible by the frame rate ({self.frame_rate})."
                )

            correct_upsample_factors = upsample_ratio
            
            if np.prod(correct_upsample_factors) != total_upsample_factor_needed:
                raise ValueError(
                    f"Product of 'upsample_ratio' from config ({np.prod(correct_upsample_factors)}) "
                    f"does not match the required upsampling factor ({total_upsample_factor_needed}) "
                    f"for the target sample rate {self.output_sample_rate} and frame rate {self.frame_rate}Hz."
                )

            print(f"Targeting {self.frame_rate}Hz -> {self.output_sample_rate}Hz. Requires {total_upsample_factor_needed}x upsampling.")
            print(f"Using DAC-style decoder with upsample factors: {correct_upsample_factors}")
            print(f"Stereo output: {stereo_output}")

            self.dac_decoder = DACStyleDecoder(
                input_channels=D,
                decoder_dim=decoder_dim,
                upsample_rates=correct_upsample_factors,
                stereo_output=stereo_output,
                stereo_hidden_dim=128,
            )

        self.last_layer_semantic = last_layer_semantic
        self.device = device

        if semantic_techer == "hubert_base_general":
            self.semantic_model = HubertModel.from_pretrained("bosonai/hubert_base", trust_remote_code=False)
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768
        else:
            raise NotImplementedError(f"Semantic teacher {semantic_techer} not implemented in this snippet.")

        self.semantic_model.eval()
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        self.semantic_downsample_factor = int(self.hop_length / (self.encoder_sample_rate / self.semantic_sample_rate) / 320)
        print(f"CALCULATED SEMANTIC DOWNSAMPLE FACTOR: {self.semantic_downsample_factor}")

        self.quantizer_dim = int((D + self.encoder_semantic_dim) // vq_scale)
        self.encoder_semantic = Encoder(input_channels=self.semantic_dim, encode_channels=self.encoder_semantic_dim)
        self.decoder_semantic = Decoder(
            code_dim=self.encoder_semantic_dim, output_channels=self.semantic_dim, decode_channels=self.semantic_dim
        )
        if isinstance(bins, int):
            self.quantizer = ResidualVectorQuantizer(dimension=self.quantizer_dim, codebook_dim=codebook_dim, n_q=n_q, bins=bins)
            self.quantizer_type = "RVQ"
        else:
            self.quantizer = ResidualFSQ(dim=self.quantizer_dim, levels=bins, num_quantizers=n_q)
            self.quantizer_type = "RFSQ"

        self.fc_prior = nn.Linear(D + self.encoder_semantic_dim, self.quantizer_dim)
        self.fc_post1 = nn.Linear(self.quantizer_dim, self.encoder_semantic_dim)
        self.fc_post2 = nn.Linear(self.quantizer_dim, D)
        
        self.downsample_mode = downsample_mode
        self.audio_tokenizer_feature_extractor = HiggsAudioFeatureExtractor(sampling_rate=self.encoder_sample_rate)
        self.apply(init_weights)

    def freeze_quantizer(self):
        for p in self.quantizer.parameters():
            p.requires_grad = False
        self._quantizer_frozen = True
        self.quantizer.eval()
        print("✓ Quantizer parameters set requires_grad=False and eval()-locked.")

    def train(self, mode: bool = True):
        super().train(mode)
        if self._quantizer_frozen:
            self.quantizer.eval()
        return self

    def freeze_for_upsampling_finetune(self):
        """Freeze encoder, semantic model, and quantizer; train full decoder (backbone + mono_head + stereo_module)."""
        print("Freezing encoder, semantic model, and quantizer for fine-tuning...")
        trainable_modules = [
            'dac_decoder',
            'decoder_semantic', 'fc_post1', 'fc_post2',
        ]
        
        for name, param in self.named_parameters():
            is_trainable = any(name.startswith(prefix) for prefix in trainable_modules)
            if not is_trainable:
                param.requires_grad = False
        self.freeze_quantizer()
        self._report_params()

    def _report_params(self):
        total_params = 0
        trainable_params = 0
        for _, p in self.named_parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        print("-----------------------------")
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"Frozen parameters: {(total_params - trainable_params) / 1e6:.2f}M")
        print("-----------------------------\n")

    @property
    def tps(self):
        return self.frame_rate

    @property
    def sampling_rate(self):
        return self.output_sample_rate

    @property
    def num_codebooks(self):
        return self.n_q

    @property
    def codebook_size(self):
        return self.quantizer_dim

    def get_last_layer(self):
        return self.decoder.layers[-1].weight

    def calculate_rec_loss(self, rec, target):
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()
        return rec_loss

    @torch.no_grad()
    def get_regress_target(self, x: torch.Tensor, orig_sr: int):
        x = torchaudio.functional.resample(x, orig_sr, self.semantic_sample_rate)
        x = x[:, 0, :]
        x = F.pad(x, (160, 160))
        target = self.semantic_model(x, output_hidden_states=True).hidden_states
        target = torch.stack(target, dim=1).mean(1)
        
        if self.downsample_mode == "step_down" and self.semantic_downsample_factor > 1:
            target = target[:, ::self.semantic_downsample_factor, :]
        
        return target

    def forward(self, x: torch.Tensor, bw: int, stereo: bool = None, width_scale: float = 1.0):
        """
        Args:
            x: Input audio (B, C, T) where C=1 for mono or C=2 for stereo
            bw: Bandwidth setting
            stereo: Override stereo output (None = use self.stereo_output)
            width_scale: Stereo width control
        """
        target_audio_length = x.shape[-1]
        
        # Handle stereo input by mixing to mono for encoder path
        if x.shape[1] == 2:
            x_mono = x.mean(dim=1, keepdim=True)
        else:
            x_mono = x

        e_semantic_input = self.get_regress_target(x_mono, orig_sr=self.output_sample_rate).detach()
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))

        model_input_audio = torchaudio.functional.resample(x_mono, self.output_sample_rate, self.encoder_sample_rate)
        e_acoustic = self.encoder(model_input_audio)
        
        min_len = min(e_acoustic.shape[2], e_semantic.shape[2])
        e_acoustic = e_acoustic[:, :, :min_len]
        e_semantic = e_semantic[:, :, :min_len]

        e = torch.cat([e_acoustic, e_semantic], dim=1)
        e = self.fc_prior(e.transpose(1, 2))

        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            quantized = quantized.transpose(1, 2)
        else:
            quantized, codes = self.quantizer(e)
            commit_loss = torch.tensor(0.0, device=x.device)

        quantized_semantic = self.fc_post1(quantized).transpose(1, 2)
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        if self.is_upsampling_model:
            # Use stereo parameter if provided, else use model default
            use_stereo = stereo if stereo is not None else self.stereo_output
            o = self.dac_decoder(quantized_acoustic, stereo=use_stereo, width_scale=width_scale)
        else:
            o = self.decoder_2(quantized_acoustic)

        o_semantic = self.decoder_semantic(quantized_semantic)
        semantic_target = e_semantic_input[:, :min_len, :].transpose(1, 2).detach()
        semantic_recon_loss = F.mse_loss(semantic_target, o_semantic)

        if o.shape[-1] != target_audio_length:
            o = o[..., :target_audio_length]
        
        return o, commit_loss, semantic_recon_loss, None

    def forward_with_intermediates(self, x: torch.Tensor, bw: int, width_scale: float = 1.0):
        """Forward pass that returns mono, side, and stereo outputs for analysis."""
        target_audio_length = x.shape[-1]
        
        if x.shape[1] == 2:
            x_mono = x.mean(dim=1, keepdim=True)
        else:
            x_mono = x

        e_semantic_input = self.get_regress_target(x_mono, orig_sr=self.output_sample_rate).detach()
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))

        model_input_audio = torchaudio.functional.resample(x_mono, self.output_sample_rate, self.encoder_sample_rate)
        e_acoustic = self.encoder(model_input_audio)
        
        min_len = min(e_acoustic.shape[2], e_semantic.shape[2])
        e_acoustic = e_acoustic[:, :, :min_len]
        e_semantic = e_semantic[:, :, :min_len]

        e = torch.cat([e_acoustic, e_semantic], dim=1)
        e = self.fc_prior(e.transpose(1, 2))

        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            quantized = quantized.transpose(1, 2)
        else:
            quantized, codes = self.quantizer(e)
            commit_loss = torch.tensor(0.0, device=x.device)

        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        if self.is_upsampling_model:
            mono, side, stereo = self.dac_decoder.forward_with_intermediates(quantized_acoustic, width_scale)
        else:
            mono = self.decoder_2(quantized_acoustic)
            side, stereo = None, None

        if mono is not None and mono.shape[-1] != target_audio_length:
            mono = mono[..., :target_audio_length]
        if side is not None and side.shape[-1] != target_audio_length:
            side = side[..., :target_audio_length]
        if stereo is not None and stereo.shape[-1] != target_audio_length:
            stereo = stereo[..., :target_audio_length]
        
        return mono, side, stereo

    def encode(self, audio_path_or_wv, sr=None, loudness_normalize=False, loudness_threshold=-23.0):
        if isinstance(audio_path_or_wv, str):
            wv, sr = librosa.load(audio_path_or_wv, mono=True, sr=None)
        else:
            wv = audio_path_or_wv
            assert sr is not None
        if loudness_normalize:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            l = meter.integrated_loudness(wv)
            wv = pyln.normalize.loudness(wv, l, loudness_threshold)
        if sr != self.encoder_sample_rate:
            wv = librosa.resample(wv, orig_sr=sr, target_sr=self.encoder_sample_rate)
        if self.audio_tokenizer_feature_extractor is not None:
            inputs = self.audio_tokenizer_feature_extractor(
                raw_audio=wv, sampling_rate=self.audio_tokenizer_feature_extractor.sampling_rate, return_tensors="pt"
            )
            input_values = inputs["input_values"].to(self.device)
        else:
            input_values = torch.from_numpy(wv).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            encoder_outputs = self._xcodec_encode(input_values)
            vq_code = encoder_outputs.audio_codes[0]
        return vq_code
    
    def _xcodec_encode(self, x: torch.Tensor, target_bw: Optional[int] = None) -> torch.Tensor:
        bw = target_bw

        e_semantic_input = self.get_regress_target(x, orig_sr=self.encoder_sample_rate).detach()
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)
        
        min_len = min(e_acoustic.shape[2], e_semantic.shape[2])
        e_acoustic = e_acoustic[:, :, :min_len]
        e_semantic = e_semantic[:, :, :min_len]

        e = torch.cat([e_acoustic, e_semantic], dim=1)
        e = self.fc_prior(e.transpose(1, 2))

        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            codes = codes.permute(1, 0, 2)
        else:
            quantized, codes = self.quantizer(e)
            codes = codes.permute(0, 2, 1)

        return EncodedResult(codes)

    def decode(self, vq_code: torch.Tensor, stereo: bool = None, width_scale: float = 1.0) -> torch.Tensor:
        """
        Decode from VQ codes to audio.
        
        Args:
            vq_code: Quantized codes
            stereo: Output stereo (None = use model default)
            width_scale: Stereo width control
        """
        if self.quantizer_type == "RVQ":
            vq_code = vq_code.permute(1, 0, 2)
            quantized = self.quantizer.decode(vq_code)
            quantized = quantized.transpose(1, 2)
        else:
            vq_code = vq_code.permute(0, 2, 1)
            quantized = self.quantizer.get_output_from_indices(vq_code)
            
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        if self.is_upsampling_model:
            use_stereo = stereo if stereo is not None else self.stereo_output
            o = self.dac_decoder(quantized_acoustic, stereo=use_stereo, width_scale=width_scale)
        else:
            o = self.decoder_2(quantized_acoustic)
            
        return o.cpu().numpy()

def prepare(checkpoint_path, config_path, device='cuda'):

    print("Loading config...")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    if 'upsample_ratio' in config:
        print("Detected upsampler fine-tuned model config. Setting encoder_sample_rate to 24000.")
        config['encoder_sample_rate'] = 24000
    else:

        config['encoder_sample_rate'] = config['sample_rate']

    print("Creating model with the following final config:")
    print(json.dumps(config, indent=2))

    model = KanadecAudioTokenizer(**config, device=device).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
  
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if unexpected_keys:
        print("\nWarning: Some keys in the checkpoint were not found in the model:")
        print(unexpected_keys)
    if missing_keys:
        print("\nWarning: Some keys in the model were not found in the checkpoint (these are randomly initialized):")
        print(missing_keys)

    model.eval()
    
    return model


def load_kanadec_audio_tokenizer(tokenizer_name_or_path, device="cuda"):
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