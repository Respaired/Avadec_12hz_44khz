---
license: mit
base_model:
- kyutai/mimi
tags:
- codec
- speech
- tokenizer
- audio_tokenizer
- speech_tokenizer
- sound
- audio
---

# What is this ?

This is a custom audio codec. <br> 

The Encoder was borrowed from `kyutai/mimi` while the decoder is trained from scratch with a latent enhancer, higher sampling rate (44.1khz) and other modifications. <br>
it should sound much better in most use cases. <br>
Backward compatible with any TTS that's trained on Mimi codes. <br>

It was trained on tens of thousands of multilingual data (English, Japanese, Persian, Russian, Arabic etc.)

# Inference

```python
import librosa
import torchaudio
from IPython.display import Audio as Sawt


from audio_processing.kanadec_audio_tokenizer import load_avadec_audio_tokenizer, encode_batch
import torch


dac_model = load_avadec_audio_tokenizer("Respair/Avadec_12hz", device='cuda')


device = 'cuda'
wav, sr = librosa.load("/home/ubuntu/kanade.mp3", sr=24000)
tensor = torch.from_numpy(wav).unsqueeze(0).to(device)

with torch.no_grad():
    codes = encode_batch(dac_model, tensor, orig_sr=24000, return_quantized=False)
    recon = dac_model.decode(codes.audio_codes.to(device))
    
Sawt(recon.squeeze(), rate=44100)
```
