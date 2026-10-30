# DeepGuard

## Current update — supplied checkpoints, 30 October 2026

**Verification completed:** 26 automated tests passed, 1 synthetic-class test skipped. Real Edge browser testing passed for upload, analysis, individual CNN outputs, desktop/mobile layout and invalid-audio handling. See [current verification](reports/CURRENT_VERIFICATION.md). The local detector is running; two-class scientific evaluation awaits the verified synthetic sample.

All three supplied checkpoints are now installed in `models/` and load strictly into the unchanged architecture. Each is 403,281 bytes, contains 30 state entries, has finite tensors and four BatchNorm counters of 65. Hashes and tensor shapes are recorded in [checkpoint audit](reports/checkpoint_audit.json). Positive counters are evidence of training-state updates, not proof of accuracy or historical provenance. Counter 65 is consistent with the notebook's later five-epoch/800-clip run (13 batches × 5 epochs); it does not match the earlier twelve-epoch/3,000-clip run. **Historical scores must not be attributed to these files.**

The existing FastAPI ensemble now produces actual predictions, including individual CNN labels/probabilities. The UI preserves its existing design and adds those individual outputs. Run `python run.py` after installing requirements, or use `Start_DeepGuard.cmd` on this computer.

### Fresh, explicitly labelled sample

The user confirmed `audio_2026-10-30_18-28-14.ogg` is human-recorded (label 0). Only its first two seconds are classified. No synthetic sample has yet been verified. Clean inference returned **SYNTHETIC / DEEPFAKE**, synthetic probability **0.702832**, genuine probability **0.297168**. This is a **false positive**, not a successful authenticity verdict.

| Condition | Synthetic probability | Prediction | Correct for human label? |
|---|---:|---|---|
| clean | 0.702832 | SYNTHETIC / DEEPFAKE | No |
| noise_20dB | 0.243202 | GENUINE HUMAN SPEECH | Yes |
| noise_10dB | 0.160994 | GENUINE HUMAN SPEECH | Yes |
| noise_0dB | 0.048063 | GENUINE HUMAN SPEECH | Yes |
| mulaw_64 | 0.780524 | SYNTHETIC / DEEPFAKE | No |
| 8khz_roundtrip | 0.577566 | SYNTHETIC / DEEPFAKE | No |


These are fresh inference results with the supplied weights. The one-sample clean accuracy is 0/1 (0%); confusion matrix, rows=true and columns=predicted genuine/synthetic, is `[[0,1],[0,0]]`. Precision, recall, F1, ROC-AUC and EER are **not reported as two-class performance** because only genuine ground truth is available. Noise variants are correlated versions of one sample and cannot establish general robustness. No source exclusion can be verified, so unseen-source evaluation remains pending.

Reproduce without training:

```powershell
pip install -r requirements-dev.txt
python -m src.evaluate --manifest demo/evaluation_manifest.json
python -m pytest -q
```

To evaluate an additional verified sample, append its path, label (0 genuine / 1 synthetic) and `label_source` to the manifest. Never infer labels from filenames. Fresh outputs go to [reports/fresh_evaluation.json](reports/fresh_evaluation.json); historical results remain in [reports/results.json](reports/results.json). The tool computes all requested metrics when both classes are present. A two-clip check still is not a representative benchmark.

## Historical finalization record (before checkpoints were supplied)

The record below preserves the earlier experiment and the initial missing-artifact state. Current model availability and fresh results are described above; historical missing-weight statements are superseded.

**Deepfake Audio Detection System — local FastAPI application around the existing three-CNN experiment.**

Status: application and historical-result deliverables are prepared. **Live audio classification is blocked because all three trained checkpoint files are missing from the accessible filesystem.** No training was performed, no weights were invented, and no replacement checkpoints were created. Feature previews operate independently of the models.

## Run locally

Use Python 3.12 (tested) or Python 3.11 in a virtual environment. From this repository:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

Open [http://localhost:8000](http://localhost:8000). No Colab, Kaggle account, public share URL, API key, Node build, or GPU is required for deployment. The service binds to the local computer only.

Recover the **original trained** `cnn_seed0.pt`, `cnn_seed1.pt`, and `cnn_seed2.pt` into `models/`, then restart the server. Alternatively set `DEEPGUARD_MODEL_DIR` to their directory before launch. The app requires all three; it never falls back to a single model or random parameters. See [models/README.md](models/README.md) for the checkpoint provenance issue. Missing files produce `model_loaded: false` and HTTP 503 for prediction.

## Problem and objective

Distinguish genuine human speech from synthetic/manipulated speech and assess generalization under unfamiliar recording conditions. The objective is research evaluation, not a universal detector or forensic authenticity certificate.

## Dataset and preparation

The supplied code uses the **Fake-or-Real (FoR)** dataset, obtained through the `mohammedabdeldayem/the-fake-or-real-dataset` mirror. The [dataset authors](https://bil.eecs.yorku.ca/datasets/) describe genuine and computer-generated speech, including separate `for-2sec` and re-recorded `for-rerec` versions. No dataset download is required for the application.

The recorded full `for-2sec` inventory is 13,956 training, 2,826 validation, and 1,088 testing WAVs, balanced by class. The evaluated ensemble uses a subset: **3,000 training / 600 validation / 600 test**, plus **600 re-recorded validation / 600 re-recorded test** clips. It takes up to 1,500/300/300 files per class from the existing directories, sorting then shuffling with `RandomState(0)` (cell 9). Class convention: **0 genuine, 1 synthetic**.

EDA evidence includes class counts, extensions, a sample's rate/duration, RMS loudness, and near-silence fractions. The source retains dataset-provided split directories; it does **not** demonstrate speaker-disjoint splitting, duplicate hashing, or source-utterance overlap checks. Leakage prevention is therefore partial, not proven. Re-recorded validation also influenced an earlier single-model selection. See [technical report](TECHNICAL_REPORT.md).

## Preprocessing and feature extraction

1. Read float32 audio with SoundFile.
2. Average channels to mono; resample with `torchaudio.functional.resample` to 16 kHz.
3. Keep the first 32,000 samples; right-pad shorter recordings with zeros.
4. `MelSpectrogram(16000, n_fft=512, hop_length=160, n_mels=64)` → `AmplitudeToDB()`.
5. Standardize each clip using its feature mean and PyTorch sample standard deviation, adding `1e-5` to the denominator.

No extra amplitude normalization, silence trimming, audio augmentation, sliding-window averaging, or denoising is introduced at inference. The plotted Log-Mel is the unstandardized dB representation of the exact model window; the CNN receives its standardized counterpart. [Torchaudio defaults](https://docs.pytorch.org/audio/2.8.0/generated/torchaudio.transforms.MelSpectrogram.html) are retained. Original training package versions were not saved; tests compare against the notebook expressions in the pinned local environment.

## CNN, baseline and ensemble

Four unchanged `Conv2d → BatchNorm2d → ReLU → MaxPool2d` blocks have channels 1→16→32→64→128, 3×3 kernels and padding 1. Adaptive average pooling, flattening, dropout 0.3, and `Linear(128, 2)` produce logits. Three seeds produce separate softmax fake probabilities; their arithmetic mean is classified as synthetic when **≥ 0.5**.

The historical Random Forest baseline has 300 trees, `random_state=0`, with two representations: 128 temporal statistics (mean/std of 64 Log-Mel bands) or 80 statistics (mean/std of 40 MFCCs). Log-Mel preserves local time-frequency structure for CNNs; pooled statistics make a compact classical-model comparison. Both original baseline implementations and outputs are preserved. No fitted RF artifact was supplied, so it is not an alternative live predictor.

**Threshold correction:** cell 12 explicitly associates the ensemble's reported metrics with 0.5. Later Gradio/report cells reused 0.871 from a different single-CNN validation experiment. This local app preserves the evaluated ensemble threshold, not that inconsistent later value.

## Completed experiment results

These are **historical saved notebook results**, unchanged and not regenerated by this application.

| Metric | In-distribution / 2-second | Re-recorded / domain shift |
|---|---:|---:|
| Accuracy | 87.0% | 51.3% |
| Precision (synthetic) | 95.1% | 56.5% |
| Recall (synthetic) | 78.0% | 11.7% |
| F1 | 0.857 | 0.193 |
| ROC-AUC | 0.955 | 0.758 |
| EER | 0.118 | 0.278 |
| TN / FP / FN / TP | 288 / 12 / 66 / 234 | 273 / 27 / 265 / 35 |

Actual matrices are available as [CSV](reports/confusion_matrix_in_distribution.csv), [re-recorded CSV](reports/confusion_matrix_rerecorded.csv), and report figures. The original output and source hashes are retained in [reports](reports/EXPERIMENT_EVIDENCE.md). ROC curve points and per-clip probability arrays were not saved; no ROC curve was invented.

## Generalization and robustness

The re-recorded experiment is a **Generalization / Domain Shift Evaluation**. It supports sensitivity to a changed acoustic channel. It does not establish a generator-held-out or speaker-held-out protocol. At threshold 0.5, 265 of 300 re-recorded synthetic clips were missed. An AUC of 0.758 does not make this operating point reliable.

| Condition | ROC-AUC | EER |
|---|---:|---:|
| Clean | 0.955 | 0.118 |
| 20 dB Gaussian noise | 0.907 | 0.167 |
| 10 dB Gaussian noise | 0.880 | 0.205 |
| 0 dB Gaussian noise | 0.798 | 0.288 |
| 8 kHz resampling round-trip | 0.813 | 0.253 |
| μ-law, 64 levels | 0.871 | 0.223 |

The “8 kHz” condition is an 8 kHz **sample rate**, implying approximately 4 kHz Nyquist bandwidth, followed by resampling to 16 kHz. μ-law is a 64-level companding/quantization test, not an MP3/AAC evaluation. All values above are from cell 12.

## Demo

Open the local URL, choose WAV/FLAC/OGG audio, inspect the full-file waveform, listen using the player, and inspect the first-two-second Log-Mel spectrogram. With valid weights loaded, click **Analyze Audio** for the label, both class probabilities, confidence, filename, original duration/rate/channels, and model input details. A model score is not calibrated certainty. Empty/corrupt files and uploads over 20 MB or 60 seconds are rejected. Silent clips receive a warning. Uploaded audio is not saved by the app.

See [demo instructions](demo/README.md) for authentic two-class testing. Do not use the software-test tone as a genuine or synthetic speech example.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Readiness, loaded model count, missing files, checkpoint hashes |
| `POST /api/features` | Multipart `file`; metadata, waveform envelope, Log-Mel matrix, warnings |
| `POST /api/predict` | Multipart `file`; all features plus ensemble verdict/probabilities |
| `GET /docs` | Interactive OpenAPI documentation |

When ready, health includes `{"status":"ok","model_loaded":true,"models":3}`. Without weights it returns HTTP 200 with `status:"degraded"`, `model_loaded:false`, `models:0`, and missing filenames. Prediction returns HTTP 503, never a fabricated probability. Upload errors use 413 (size) or 422 (invalid audio). `duration_seconds`, `sample_rate`, and `channels` describe the original file; `processed_*` describe the 2-second/16-kHz/mono input. `confidence` is the larger class probability.

```powershell
curl.exe http://localhost:8000/api/health
curl.exe -F "file=@demo/genuine.wav" http://localhost:8000/api/predict
```

## Repository and tests

`app/` contains FastAPI and the static UI; `src/` holds inference; `models/` expects recovered artifacts; `research/` preserves original source without importing it; `reports/` contains historical evidence and verification; `tests/` checks preprocessing, API validation, threshold behavior and checkpoint absence; `demo/` documents two-class testing.

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

The real-checkpoint tests deliberately skip until original models and labelled test audio are available. A skipped integration test is not proof of working classification. [Verification report](reports/VERIFICATION.md) records actual checks and blockers.

## Limitations and future work

Missing original model artifacts prevent live classification and matching weights to the saved scores. The notebook contains both a random-weight overwrite cell and a later five-epoch retraining under the same filenames; neither establishes continuity with the 12-epoch evaluated ensemble. A trained-state check can reject obvious initialization, but cannot prove historical provenance.

The detector is sensitive to recording channels, noise and quantization. No dedicated speaker-identity ablation, generator-held-out evaluation, duplicate audit, calibrated probabilities, demographic performance breakdown, or forensic validation is evidenced. Loudness/silence shortcut statistics are present, but do not prove the CNN's causal dependence. Future work: recover provenance-linked artifacts, audit source/speaker splits, perform generator-held-out evaluation, controlled speaker/channel ablations, and validation-only calibration. Retraining is outside this finalization task.

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md), [PANEL_QA.md](PANEL_QA.md), and [TASK_COMPLIANCE.md](TASK_COMPLIANCE.md). This is a GitHub-ready local repository; no remote repository has been published.
