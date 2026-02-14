---
license: cc-by-4.0
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
language:
- en
- fa
- ja
- ar
- ru
---

# What is this ?

This is a custom speech tokenizer. <br> 

- The Encoder was borrowed from `kyutai/mimi` while the decoder is trained from scratch using a different architecrue, higher sampling rate (44.1khz) and other modifications. <br>
it should sound much better in most use cases. <br>

- Backward compatible with any TTS that's trained on Mimi codes. <br>

- It was trained on tens of thousands of multilingual data (English, Japanese, Persian, Russian, Arabic etc.)

# Inference

```python
import librosa
import torchaudio
from IPython.display import Audio as Sawt


from audio_processing.kanadec_audio_tokenizer import load_avadec_audio_tokenizer, encode_batch
import torch


dac_model = load_avadec_audio_tokenizer("Respair/Avadec_12hz_44khz", device='cuda')


device = 'cuda'
wav, sr = librosa.load("path_to/audio.mp3", sr=24000)
tensor = torch.from_numpy(wav).unsqueeze(0).to(device)

with torch.no_grad():
    encoded = encode_batch(dac_model, tensor, orig_sr=24000, return_quantized=False)
    recon = dac_model.decode(encoded.audio_codes.to(device))
    
Sawt(recon.squeeze(), rate=44100)
```
