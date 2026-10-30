# DeepGuard — compliance with the supplied assignment

Reference: `research/assignment-current.jpg`, “Special Task: Deepfake Audio Detection.” The sales-forecasting text above that heading belongs to another task.

The local application is functional with three supplied CNN checkpoints. **The assignment is not fully evidenced:** current two-class evaluation, source exclusion, leakage auditing and a published GitHub repository remain incomplete. Historical notebook experiments must not be treated as results of the currently supplied weights.

## 1. Dataset preparation

| Requirement in image | Status | Evidence / remaining work |
|---|---|---|
| Public dataset containing genuine and synthetic speech | Historical evidence available | Fake-or-Real dataset; notebook cell 2 and technical report §4. |
| Audio preprocessing and exploratory data analysis | Implemented; historical EDA available | Mono/16-kHz/two-second pipeline; class counts, sample format, RMS and silence statistics in cells 2 and 10. |
| Appropriate train/validation/test splits | Historical evidence available | Cell 9 records 3000/600/600 clips for the earlier experiment; not proof of the later supplied checkpoints' exact split membership. |
| Prevent data leakage | PARTIAL | Separate directories retained. Speaker/source overlap and duplicate hashes were not independently audited. |

## 2. Feature extraction

| Requirement in image | Status | Evidence / remaining work |
|---|---|---|
| Log-Mel spectrograms | VERIFIED | src/audio.py, numerical parity tests and browser visualization; 64 Mel bands. |
| Additional representation and comparison | Historical evidence available | 40 MFCCs with temporal mean/std versus Log-Mel statistics using Random Forest; cell 10, report §§7–8. |

## 3. Model development

| Requirement in image | Status | Evidence / remaining work |
|---|---|---|
| Baseline classifier | Historical implementation/results available | 300-tree Random Forest. Fitted RF artifact not supplied; no retraining performed. |
| CNN audio classifier | VERIFIED for local inference | Three compatible state dictionaries loaded; original architecture retained; reports/checkpoint_audit.json. |
| Compare CNN with an alternative | Historical comparison available | RF versus CNN/ensemble tables. Current weights have not been freshly compared on a common labelled test set. |
| Explain architecture and decisions | DOCUMENTED | Technical report §§7–10 and panel Q&A. |

## 4. Model evaluation

Synthetic is class 1; genuine is class 0. Threshold: 0.5. Historical values remain in reports/results.json; current results are in reports/fresh_evaluation.json.

| Requirement in image | Historical experiment | Currently supplied checkpoints |
|---|---|---|
| Precision and recall | Actual cell-12 values preserved | PENDING verified synthetic samples; one genuine clip cannot support two-class evaluation. |
| F1-score | Actual cell-12 values preserved | PENDING; not fabricated. |
| ROC-AUC | 0.955 in-domain / 0.758 re-recorded | PENDING both classes. |
| Equal Error Rate | 0.118 in-domain / 0.278 re-recorded | PENDING both classes. |
| Confusion matrix | Actual matrices exported to CSV/JSON/figures | One-clip matrix [[0,1],[0,0]]; rows=true, columns=predicted genuine/synthetic. |
| False-positive and false-negative analysis | Historical aggregate analysis documented | Confirmed genuine clip is a false positive, P(synthetic)=0.702832. No synthetic ground truth yet to assess false negatives. |

Current clean accuracy is 0/1 on one user-confirmed genuine recording. This is a smoke-test outcome, not a benchmark. src/evaluate.py computes all requested metrics once both verified classes are supplied; a two-clip check still would not establish representative performance.

## 5. Generalization and robustness

| Requirement in image | Status | Evidence / remaining work |
|---|---|---|
| Source or speech generator absent from training | NOT ESTABLISHED | Historical re-recording shows acoustic domain shift, not verified generator exclusion. Current sample has no documented source exclusion. |
| Background noise and audio compression | Historical study plus fresh limited check | Historical 20/10/0-dB noise, μ-law and 8-kHz round-trip results retained. All conditions also executed on the current genuine sample; one recording is not a robustness benchmark. |
| Speaker identity or unintended dataset patterns | PARTIAL | RMS/silence associations examined. Dedicated speaker ablation and causal shortcut tests were not performed. |

## 6. Deliverables

| Deliverable in image | Status | Location |
|---|---|---|
| GitHub repository with complete implementation | Local repository prepared; NOT published | Project source and ZIP; no remote URL. |
| README with setup/execution | COMPLETE | README.md, run.py, Start_DeepGuard.cmd. |
| Dataset description and preprocessing | COMPLETE as documentation | README and technical report §§4–7; raw dataset not bundled. |
| Trained models and evaluation results | Models supplied; evaluation PARTIAL | Three .pt files with hash/state audit; separate historical and current results. |
| Technical report comparing approaches and limitations | COMPLETE as documentation | TECHNICAL_REPORT.md; current/historical evidence distinguished. |
| Genuine/synthetic prediction demonstration | VERIFIED operationally | FastAPI/browser, both probabilities, confidence, individual CNN predictions, waveform and Log-Mel. Operational success does not imply accurate classification. |

## Verification and remaining work

**26 automated tests passed, 1 skipped; 12 browser checks passed.** The skipped test awaits the user's separately verified synthetic sample. No retraining occurred; weights and threshold were not tuned to the genuine test clip.

Remaining evidence: verified genuine/synthetic test samples with provenance; leakage/source/speaker separation audit; a source-held-out evaluation and controlled shortcut analysis; publication to the intended GitHub repository. None can be replaced with invented results or an unqualified claim of completion.

See [current verification](reports/CURRENT_VERIFICATION.md), [technical report](TECHNICAL_REPORT.md), and [historical evidence](reports/EXPERIMENT_EVIDENCE.md).
