"""Requires actual recovered artifacts. Never substitutes a model or fabricates weights."""
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from src.model import NAMES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("audio_variable", ["DEEPGUARD_GENUINE_AUDIO", "DEEPGUARD_SYNTHETIC_AUDIO"])
def test_authentic_audio_inference(audio_variable):
    directory = Path(os.environ.get("DEEPGUARD_MODEL_DIR", ROOT / "models"))
    if any(not (directory/name).is_file() for name in NAMES):
        pytest.skip("Original three trained checkpoints unavailable; real inference is blocked.")
    path = os.environ.get(audio_variable)
    if not path or not Path(path).is_file():
        pytest.skip(f"Authentic labelled test audio unavailable: {audio_variable}")
    with TestClient(create_app(directory)) as client:
        health = client.get("/api/health").json()
        assert health["model_loaded"], health["detail"]
        with open(path, "rb") as f:
            response = client.post("/api/predict", files={"file": (Path(path).name, f, "audio/wav")})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["models"] == 3 and len(data["seed_probabilities"]) == 3
        assert data["fake_probability"] == pytest.approx(sum(data["seed_probabilities"])/3)
        assert data["label"] == ("SYNTHETIC / DEEPFAKE" if data["fake_probability"] >= 0.5 else "GENUINE HUMAN SPEECH")
        # This is a pipeline check, not a claim that every labelled sample must be classified correctly.
