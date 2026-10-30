# Current DeepGuard verification — 30 October 2026

Latest maintenance check: fixed the missing favicon, added readable network/invalid-response errors and request timeouts, enabled retrying the same upload, and added a portable Python fallback to the Windows launcher. Re-ran the suite: **26 passed, 1 skipped** (4.44 seconds). All **12 browser checks passed**, including simulated network failure and successful same-file retry. Server restarted successfully on port 8000. The known false positive remains a model limitation; weights and threshold were not changed.

This report supersedes the missing-model and blocked-browser status in the initial `VERIFICATION.md`.

The existing application was copied into the newly selected empty workspace and extended, not rebuilt. Original checkpoint downloads were preserved. All three supplied state dictionaries load strictly into the unchanged four-block CNN architecture. Each checkpoint is 403,281 bytes with 30 keys, finite values, distinct SHA-256 hash and BatchNorm counters of 65. See `checkpoint_audit.json` for tensor shapes and hashes. These counters are consistent with the notebook's later five-epoch run; continuity with the historical twelve-epoch ensemble is not established.

## What was executed

- All automated Python tests: **26 passed, 1 skipped**, one upstream deprecation warning. Full output: `current_tests.txt`. The skipped test requires a verified synthetic recording.
- Real browser check using headless Edge against the live FastAPI service: passed. Uploaded the user-confirmed human OGG, clicked Analyze Audio, verified actual probabilities and three individual predictions, checked a 390-pixel mobile viewport for horizontal overflow, and verified invalid audio clears the previous verdict and disables analysis. No JavaScript page errors. Evidence: `current_browser_verification.json` and `figures/current-desktop.png`, `figures/current-mobile.png`.
- Actual frontend JavaScript consuming live API responses in a minimal DOM harness: passed. Evidence: `frontend_live_contract.json`. This is additional to, not a substitute for, the real browser check.
- Inference-only evaluation of clean audio, Gaussian noise at 20/10/0 dB SNR, 64-level μ-law and 16→8→16-kHz resampling: executed. Results: `fresh_evaluation.json`.
- Exact OGG decoding parity test added: upload decoding must equal the research `sf.read` pipeline. Explicit `seek(0)` fixes a libsndfile Opus decoder-position discrepancy detected during live testing.

## Actual result, not a success claim about accuracy

The user explicitly labelled `audio_2026-10-30_18-28-14.ogg` as genuine. The model predicts synthetic with probability **0.7028319835662842** and genuine with **0.2971680164337158**. The three synthetic probabilities are approximately 0.750000, 0.626565 and 0.731931. All three are wrong for this sample at threshold 0.5. The UI correctly displays the real model output.

Clean confusion matrix (true rows, predicted columns, genuine/synthetic order): `[[0,1],[0,0]]`. Accuracy on this one clip is 0/1. The noise conditions change the prediction to genuine; μ-law and bandwidth limitation retain a synthetic verdict. These transformations of a single recording do not constitute a representative robustness benchmark.

There is no verified synthetic sample yet. Two-class precision/recall/F1/ROC-AUC/EER are therefore not presented as measured detector performance. The evaluator reports unavailable discrimination metrics as null and computes them when both verified classes are present. Unseen-source exclusion is not established, so that experiment remains pending. Historical metrics stay separate and are not attributed to these supplied weights.

## Run and reproduce

The server is running at `http://localhost:8000`. `Start_DeepGuard.cmd` launches with the existing installed environment on this machine; standard portable setup remains `pip install -r requirements.txt` then `python run.py`.

```powershell
pip install -r requirements-dev.txt
$env:DEEPGUARD_GENUINE_AUDIO = (Resolve-Path demo/audio_2026-10-30_18-28-14.ogg).Path
python -m pytest -q
python -m src.evaluate --manifest demo/evaluation_manifest.json
node tests/frontend_contract.cjs
```

Add the separately verified synthetic file to the evaluation manifest with label 1 and its provenance when available. Do not infer its class from a name. No training, random checkpoint creation, or metric fabrication occurred.
