"""The supplied notebook's waveform and Log-Mel preprocessing, on CPU."""
from io import BytesIO
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import nn

SAMPLE_RATE = 16000
SAMPLES = 32000
MAX_BYTES = 20 * 1024 * 1024


def prepare_waveform(audio: np.ndarray, sample_rate: int) -> torch.Tensor:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        a = a.mean(1)
    if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError("Audio must contain finite, nonempty samples.")
    if sample_rate != SAMPLE_RATE:
        a = torchaudio.functional.resample(torch.tensor(a), sample_rate, SAMPLE_RATE).numpy()
    a = a[:SAMPLES]
    if len(a) < SAMPLES:
        a = np.pad(a, (0, SAMPLES - len(a)))
    return torch.tensor(a).unsqueeze(0)


class FeatureExtractor:
    def __init__(self):
        self.mel = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(16000, n_fft=512, hop_length=160, n_mels=64),
            torchaudio.transforms.AmplitudeToDB(),
        )

    @torch.inference_mode()
    def __call__(self, waveform):
        f = self.mel(waveform).unsqueeze(1)
        normalized = (f - f.mean((1, 2, 3), keepdim=True)) / (
            f.std((1, 2, 3), keepdim=True) + 1e-5
        )
        if not torch.isfinite(normalized).all():
            raise ValueError("Audio produced non-finite features.")
        return f, normalized


def envelope(audio, bins=900):
    chunks = np.array_split(audio, min(bins, len(audio)))
    return {"min": [float(x.min()) for x in chunks], "max": [float(x.max()) for x in chunks]}


def decode_audio(data: bytes, filename: str, extractor: FeatureExtractor):
    if not data:
        raise ValueError("The audio file is empty.")
    if len(data) > MAX_BYTES:
        raise ValueError("Audio upload exceeds 20 MB.")
    try:
        with sf.SoundFile(BytesIO(data)) as source:
            sr, channels, frames = source.samplerate, source.channels, source.frames
            if sr < 1000 or sr > 192000 or channels > 8 or frames <= 0 or frames / sr > 60:
                raise ValueError("Use nonempty audio up to 60 seconds, 8 channels, and 192 kHz.")
            # sf.read() explicitly seeks to frame zero. Opus decoding can otherwise
            # start at a different decoder position; preserve research-time behavior.
            source.seek(0)
            a = source.read(dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError("Cannot decode this audio. Use a valid WAV, FLAC, or OGG file.") from exc
    waveform = prepare_waveform(a, sr)
    logmel, features = extractor(waveform)
    original_mono = a.mean(1)
    metadata = {
        "filename": filename.replace("\\", "/").split("/")[-1][:240],
        "duration_seconds": frames / sr,
        "sample_rate": sr,
        "channels": channels,
        "processed_duration_seconds": 2.0,
        "processed_sample_rate": SAMPLE_RATE,
        "processed_channels": 1,
        "analyzed_source_seconds": min(frames / sr, 2.0),
        "waveform": envelope(original_mono),
        "processed_waveform": envelope(waveform[0].numpy()),
        "log_mel": logmel[0, 0].tolist(),
        "mel_bands": 64,
        "hop_seconds": 0.01,
        "warnings": [],
    }
    if frames / sr > 2:
        metadata["warnings"].append("Only the first 2 seconds are classified, matching training.")
    elif frames / sr < 2:
        metadata["warnings"].append("Audio is zero-padded to 2 seconds, matching training.")
    if float(np.max(np.abs(original_mono))) < 1e-5:
        metadata["warnings"].append("This clip is silent or nearly silent; a speech verdict is unreliable.")
    return metadata, features
