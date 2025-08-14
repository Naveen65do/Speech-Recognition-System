import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio(audio_path: str):
    speech, sr = torchaudio.load(audio_path)
    
    # Convert stereo to mono
    if speech.shape[0] > 1:
        speech = torch.mean(speech, dim=0, keepdim=True)
    
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    speech = resampler(speech)
    
    return speech.squeeze()


def transcribe_with_whisper(audio_path: str, model_name="openai/whisper-tiny", language="english") -> str:
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    speech = load_audio(audio_path)
    input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription