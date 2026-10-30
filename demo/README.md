# Current status update

The three supplied models now load and produce real predictions. A user-confirmed human clip was tested and falsely labelled synthetic. Synthetic-class and verified unseen-source tests are pending. See the current README/technical report and reports/fresh_evaluation.json. The earlier missing-model statements and checklist counts below are historical and superseded; they are not the current readiness report.

# Panel demo

1. Install dependencies and recover all three original trained models as described in the main README.
2. Run `python run.py`; open `http://localhost:8000`.
3. Confirm the detector reports three loaded checkpoints and `/api/health` says `model_loaded: true`.
4. Select a documented genuine speech file from the original `for-2sec/for-2seconds/testing/real` split. Check the original filename, duration, rate and channels; listen; inspect waveform and Log-Mel; click Analyze Audio. Record both probabilities and the predicted class without assuming correctness.
5. Repeat with documented synthetic speech from `testing/fake`. A detector can misclassify either class. Preserve ground-truth labels separately from predictions.
6. Discuss the historical confusion matrices and poor re-recorded recall, rather than using two selected clips as an accuracy estimate.

No genuine/synthetic dataset audio was found locally during the C:\ and D:\ search. No speech samples have been invented or relabelled. These two tests are **blocked**, not passed. Feature and UI checks use an explicitly labelled non-speech test tone generated in the temporary work directory; it is not a model-quality experiment.

When authentic files are recovered, optional integration tests use:

```powershell
$env:DEEPGUARD_GENUINE_AUDIO = 'C:\path\to\testing\real\sample.wav'
$env:DEEPGUARD_SYNTHETIC_AUDIO = 'C:\path\to\testing\fake\sample.wav'
python -m pytest -q tests/test_real_inference.py
```

If weights are missing, feature previews remain available, Analyze Audio stays disabled, and the API refuses prediction. Do not present this state as a working classifier.
