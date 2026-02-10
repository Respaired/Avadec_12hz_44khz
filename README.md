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
This is a custom audio codec. <br> 

the Encoder was borrowed from `Mimi-12.hz - 32cb` while the decoder is trained from scratch with a latent enhancer, higher sampling rate (44.1khz) and other modifications. <br>
it should sound much better in most use cases. <br>
Backward compatible with any TTS that's trained on Mimi codes. 

It was trained on tens of thousands of multilingual data (English, Japanese, Persian, Russian, Arabic etc.)