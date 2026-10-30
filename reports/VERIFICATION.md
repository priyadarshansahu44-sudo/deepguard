# Execution and verification report

Date: 30 October 2026 (Asia/Calcutta). No training, checkpoint creation or checkpoint replacement occurred.

## Actual execution

Installed `requirements-dev.txt` (which includes `requirements.txt`) in a local Python 3.12.14 virtual environment. The exact resolved packages are recorded in [environment-lock.txt](environment-lock.txt). The machine's default `python` alias is a Windows Store launcher, so testing used the bundled Python executable and an isolated environment under the task's `work/` directory. `Start_DeepGuard.cmd` can use that existing environment on this computer; a portable fresh setup is documented in the README.

The environment's Python 0700 temporary-directory ACL handling initially prevented pip writes. Installation succeeded using inherited ACLs for temporary folders confined to `work/`; no system ACL or security setting was changed. Dependency downloads required several network resumes and completed successfully.

Executed `python run.py` from the repository. Uvicorn reported:

```text
Started server process [18808]
Waiting for application startup.
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

## Automated software tests

Command: `python -m pytest -q -p no:cacheprovider`.

```text
21 passed, 2 skipped, 1 warning in 10.42s
```

The warning is a Starlette/AnyIO deprecated alias; no test failed. Tests cover exact Log-Mel numerical parity with safely extracted original notebook definitions, exact architecture parity on meta tensors, mono/resampling/crop/pad behavior, silence and malformed input, original-vs-processed metadata, waveform/spectrogram dimensions, probability averaging, threshold equality, missing-model refusal, API/static assets, upload limits, and consistency of historical counts and rounded metrics.

Only the selected pure feature definitions and meta architecture definitions were executed from the source. No notebook top-level statements, training loop, data download or checkpoint save was executed.

**Two skipped tests:** genuine speech inference and synthetic speech inference with original trained checkpoints. They explicitly skip when authentic artifacts are absent. No mocked network is used to claim model success.

## Live HTTP verification

Seven additional checks passed against the running Uvicorn process, not only an in-process test client. [Machine-readable results](live_api_verification.json) include:

| Boundary | Status | Evidence |
|---|---|---|
| GET / | PASS | HTTP 200; DeepGuard HTML |
| Static assets | PASS | JavaScript and CSS served successfully |
| GET /api/health | PASS | HTTP 200; degraded, model_loaded=false, models=0, all three missing names |
| Upload → decode → preprocessing → features | PASS | Actual 3-second, 44.1-kHz stereo upload returns original metadata and a 64×201 Log-Mel matrix |
| Missing weights → prediction response | PASS | POST /api/predict returns 503 with no fabricated probability |
| Invalid upload | PASS | Corrupt WAV content returns 422 |
| API documentation | PASS | GET /docs returns 200 |

The upload fixture is a generated non-speech software-test tone. It is explicitly not a genuine or synthetic speech sample and was not used for an accuracy claim. The app's feature computation is verified; a detector prediction has not been produced.

## Browser and visual verification

The local URL was requested in the app browser panel; the tool reported `queued`, which does not confirm rendering. Automated browser verification was attempted with the bundled Playwright package. The bundled headless Chromium executable was missing. The installed Edge launch then failed with `browserType.launch: spawn EPERM`. No browser controls, security settings or system permissions were bypassed.

JavaScript syntax checking (`node --check app/static/app.js`) passed. **Actual browser rendering, mobile layout, canvas rendering and user interaction remain unverified.** No UI screenshots or successful browser test counts are claimed. The intended browser test flow is documented in `demo/README.md`.

The historical confusion-matrix and robustness figures were rendered with Matplotlib and visually inspected. Labels and numerical values match the extracted notebook output.

## Artifact search and model boundary

The entire accessible C:\ and D:\ filesystem was enumerated with `rg --files --hidden --no-ignore`, using filters for named checkpoint variants, related checkpoints/archives and WAV/FLAC/OGG files. Earlier targeted searches covered Downloads (including Telegram Desktop), Desktop, Documents and the workspace. The broad search returned 440 audio/archive filename matches and no relevant checkpoint or identifiable FoR test-directory match. It reported 612 access errors, recorded in [FILE_SEARCH.json](FILE_SEARCH.json). Those counts include unrelated system/application audio and inaccessible system/cache directories.

Missing files: **cnn_seed0.pt, cnn_seed1.pt, cnn_seed2.pt**. No candidates were available for size/state-dictionary inspection. Files in inaccessible folders, unexpanded archives, offline cloud locations, remote Colab/Kaggle sessions, and unfollowed junction/symlink targets cannot be ruled out. The search is not a claim that the files do not exist anywhere.

The first broken model boundary is checkpoint loading. Therefore the three-CNN forward passes, authentic two-class API results, and frontend prediction display cannot be verified. The required resolution is recovery of the original evaluated weights and authentic test audio, followed by inference-only checks. Retraining or random reconstruction would not resolve original-artifact provenance.

## Repository status

A local Git repository was initialized. Source, documentation, original research, historical evidence and tests are prepared for GitHub. No remote repository was created or published. A ZIP contains deliverable files without the virtual environment, Git internals or bytecode caches.

Overall: **local service and feature API verified; complete detector demonstration blocked; browser visual QA incomplete.**
