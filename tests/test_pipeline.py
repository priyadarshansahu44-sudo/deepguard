"""Software checks only; test signals are not labelled speech or model evidence."""
import ast
from io import BytesIO
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio
from torch import nn
from fastapi.testclient import TestClient
from src.audio import FeatureExtractor, prepare_waveform, decode_audio, MAX_BYTES
from src.model import Ensemble, architecture, summarize
from app.main import create_app

ROOT = Path(__file__).resolve().parents[1]


def wav(seconds=1.0, rate=16000, channels=1):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    a = 0.1 * np.sin(2 * np.pi * 440 * t)
    if channels > 1:
        a = np.stack([a] * channels, axis=1)
    f = BytesIO(); sf.write(f, a, rate, format="WAV", subtype="FLOAT")
    return f.getvalue()


@pytest.fixture(scope="module")
def client():
    # Nonexistent directory: never fabricate checkpoint files for the tests.
    with TestClient(create_app(ROOT / "tests" / "no-checkpoints-present")) as c:
        yield c


def test_exact_notebook_feature_parity():
    notebook = json.loads((ROOT / "research/original.ipynb").read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][9]["source"]))
    # Execute only the original mel assignment and pure feats function. No source
    # data loading, training, checkpoint write, or top-level notebook code runs.
    nodes = [n for n in tree.body if (
        isinstance(n, ast.FunctionDef) and n.name == "feats") or (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "mel" for t in n.targets))]
    assert len(nodes) == 2
    scope = {"torch": torch, "nn": nn, "torchaudio": torchaudio, "dev": "cpu"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook_feature_reference", "exec"), scope)
    t = torch.arange(32000, dtype=torch.float32) / 16000
    x = torch.stack([0.1 * torch.sin(2 * torch.pi * 440 * t), torch.zeros_like(t)])
    _, actual = FeatureExtractor()(x)
    expected = scope["feats"](x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 1, 64, 201)


def test_stereo_resample_crop_matches_notebook():
    a = np.linspace(-0.2, 0.2, 44100 * 3, dtype=np.float32)
    stereo = np.stack([a, a * 0.3], axis=1)
    expected = torchaudio.functional.resample(torch.tensor(stereo.mean(1)), 44100, 16000)[:32000]
    torch.testing.assert_close(prepare_waveform(stereo, 44100)[0], expected, rtol=0, atol=0)


def test_right_padding_preserves_input():
    a = np.full(8000, 0.2, dtype=np.float32)
    actual = prepare_waveform(a, 16000)[0]
    assert actual.shape == (32000,)
    np.testing.assert_array_equal(actual[:8000].numpy(), a)
    assert torch.count_nonzero(actual[8000:]) == 0


@pytest.mark.parametrize("a", [np.array([], dtype=np.float32), np.array([np.nan]), np.array([np.inf])])
def test_invalid_samples_rejected(a):
    with pytest.raises(ValueError): prepare_waveform(a, 16000)


def test_exact_architecture_without_initializing_weights():
    notebook = json.loads((ROOT / "research/original.ipynb").read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][9]["source"]))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "block"]
    nodes += [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "model" for t in n.targets)]
    assert len(nodes) == 2
    scope = {"nn": nn, "dev": "meta"}
    with torch.device("meta"):
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook_architecture_reference", "exec"), scope)
    actual = architecture()
    assert repr(actual) == repr(scope["model"])
    assert {k: tuple(t.shape) for k,t in actual.state_dict().items()} == {k: tuple(t.shape) for k,t in scope["model"].state_dict().items()}
    assert all(t.is_meta for t in actual.state_dict().values())


@pytest.mark.parametrize("p,label", [([0.2,0.4,0.6], "GENUINE HUMAN SPEECH"), ([0.4,0.5,0.6], "SYNTHETIC / DEEPFAKE")])
def test_probability_average_and_exact_half_threshold(p, label):
    result = summarize(p)
    assert result["fake_probability"] == pytest.approx(sum(p)/3)
    assert result["label"] == label
    assert result["genuine_probability"] + result["fake_probability"] == pytest.approx(1)


@pytest.mark.parametrize("p", [[0.1], [0.1,0.2], [0.1,0.2,float('nan')], [0.1,0.2,1.1]])
def test_invalid_ensemble_probabilities(p):
    with pytest.raises(ValueError): summarize(p)


def test_no_checkpoint_fallback(client):
    h = client.get("/api/health").json()
    assert h["status"] == "degraded" and h["model_loaded"] is False and h["models"] == 0
    assert h["missing_models"] == [f"cnn_seed{i}.pt" for i in range(3)]
    response = client.post("/api/predict", files={"file": ("tone.wav", wav(), "audio/wav")})
    assert response.status_code == 503
    assert "fake_probability" not in response.json()


def test_static_and_openapi(client):
    assert client.get("/").status_code == 200
    assert "DeepGuard" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/favicon.ico").status_code == 200
    assert "/api/predict" in client.get("/openapi.json").json()["paths"]


def test_feature_api_metadata_and_visualization(client):
    response = client.post("/api/features", files={"file": ("test-tone.wav", wav(3, 44100, 2), "audio/wav")})
    assert response.status_code == 200
    data = response.json()
    assert data["sample_rate"] == 44100 and data["channels"] == 2
    assert data["duration_seconds"] == 3 and data["processed_duration_seconds"] == 2
    assert len(data["log_mel"]) == 64 and len(data["log_mel"][0]) == 201
    assert len(data["waveform"]["min"]) == 900
    assert data["warnings"] and "fake_probability" not in data


@pytest.mark.parametrize("data", [b"", b"not a real audio file"])
def test_bad_upload_rejected(client, data):
    assert client.post("/api/features", files={"file": ("bad.wav", data)}).status_code == 422


def test_upload_limits(client):
    assert client.post("/api/features", files={"file": ("huge.wav", b"x" * (MAX_BYTES+1))}).status_code == 413
    assert client.post("/api/features", files={"file": ("long.wav", wav(61, 8000))}).status_code == 422


def test_silence_warning_and_finite_features():
    f = BytesIO(); sf.write(f, np.zeros(32000), 16000, format="WAV")
    data, features = decode_audio(f.getvalue(), "silent.wav", FeatureExtractor())
    assert any("silent" in x for x in data["warnings"])
    assert torch.isfinite(features).all()


def test_historical_results_match_saved_counts():
    data = json.loads((ROOT / "reports/results.json").read_text())
    for metrics in data["evaluations"].values():
        tn, fp, fn, tp = (metrics[k] for k in ["TN", "FP", "FN", "TP"])
        assert round((tp+tn)/(tp+tn+fp+fn),3) == metrics["accuracy"]
        assert round(tp/(tp+fp),3) == metrics["precision"]
        assert round(tp/(tp+fn),3) == metrics["recall"]
        assert round(2*tp/(2*tp+fp+fn),3) == metrics["f1"]
    assert data["threshold"] == 0.5
