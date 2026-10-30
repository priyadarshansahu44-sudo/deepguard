"""Exact notebook architecture; require all three existing trained checkpoints."""
from pathlib import Path
import hashlib
import torch
from torch import nn

THRESHOLD = 0.5  # Notebook cell 12: threshold associated with the reported ensemble metrics.
NAMES = tuple(f"cnn_seed{i}.pt" for i in range(3))


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


def architecture():
    # Meta tensors allocate no parameter data. Every parameter must be supplied by the checkpoint.
    with torch.device("meta"):
        return nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 2))


def summarize(probabilities):
    if len(probabilities) != 3:
        raise ValueError("Exactly three CNN probabilities are required.")
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    if not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid model probabilities.")
    fake = float(p.mean())
    return {
        "label": "SYNTHETIC / DEEPFAKE" if fake >= THRESHOLD else "GENUINE HUMAN SPEECH",
        "fake_probability": fake,
        "genuine_probability": 1 - fake,
        "confidence": max(fake, 1 - fake),
        "threshold": THRESHOLD,
        "seed_probabilities": [float(v) for v in p],
        "individual_predictions": [
            {"seed": i, "fake_probability": float(v), "genuine_probability": 1-float(v),
             "label": "SYNTHETIC / DEEPFAKE" if v >= THRESHOLD else "GENUINE HUMAN SPEECH"}
            for i, v in enumerate(p)
        ],
        "models": 3,
    }


class Ensemble:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.models = []
        self.hashes = {}
        self.error = None
        self.missing = [name for name in NAMES if not (self.directory / name).is_file()]
        if self.missing:
            self.error = "Missing original checkpoints: " + ", ".join(self.missing)
            return
        try:
            loaded = []
            for name in NAMES:
                path = self.directory / name
                if path.stat().st_size > 10 * 1024 * 1024:
                    raise ValueError(f"Unexpected checkpoint size: {name}")
                state = torch.load(path, map_location="cpu", weights_only=True)
                model = architecture()
                model.load_state_dict(state, strict=True, assign=True)
                if any(t.is_meta for t in model.state_dict().values()):
                    raise ValueError(f"Incomplete checkpoint: {name}")
                if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
                    raise ValueError(f"Non-finite checkpoint values: {name}")
                # Cell 19 saved randomly initialized networks. Their BatchNorm counters are zero.
                if any(int(b.num_batches_tracked) <= 0 for b in model.modules() if isinstance(b, nn.BatchNorm2d)):
                    raise ValueError(f"{name} has untrained BatchNorm counters; recover trained weights.")
                model.eval()
                model.requires_grad_(False)
                loaded.append(model)
                self.hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.models = loaded
        except Exception as exc:
            self.models = []
            self.hashes = {}
            self.error = f"Checkpoint validation failed: {exc}"

    @property
    def ready(self):
        return len(self.models) == 3

    @torch.inference_mode()
    def predict(self, features):
        if not self.ready:
            raise RuntimeError(self.error or "The three trained models are unavailable.")
        probs = [torch.softmax(m(features), 1)[:, 1].item() for m in self.models]
        return summarize(probs)
