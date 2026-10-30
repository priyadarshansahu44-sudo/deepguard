# Current status update

The three supplied models now load and produce real predictions. A user-confirmed human clip was tested and falsely labelled synthetic. Synthetic-class and verified unseen-source tests are pending. See the current README/technical report and reports/fresh_evaluation.json. The earlier missing-model statements and checklist counts below are historical and superseded; they are not the current readiness report.

# DeepGuard — panel questions and honest answers

**Why deepfake audio detection?**
Synthetic speech can imitate natural speech, so an acoustic classifier can help flag suspicious samples. It is one piece of evidence, not an authenticity guarantee. My project also measures how much its performance deteriorates outside clean matched conditions.

**Why Log-Mel spectrograms?**
They represent energy over time and perceptually spaced frequency bands. A 2D CNN can examine local time-frequency patterns. I use the existing experiment's 64 Mel bands, 512-point FFT and 160-sample hop, followed by dB conversion and per-clip standardization.

**Why a CNN?**
Convolutions learn local patterns shared across a spectrogram. My architecture has four convolution/BatchNorm/ReLU/pooling blocks, then global average pooling, dropout and a two-class head. I did not replace the trained architecture during deployment.

**Why Random Forest?**
It provides a classical baseline on compact acoustic statistics. The existing experiment compares 300-tree RF models on Log-Mel and MFCC temporal means/standard deviations. This helps judge whether the CNN adds value over simpler representations/classifiers. No fitted RF artifact was supplied for a live baseline demo.

**Why an ensemble?**
I average the fake probabilities of three independently seeded CNNs. The historical in-domain AUC rises from individual values 0.927, 0.930 and 0.948 to 0.955. It combines their predictions but does not solve the generalization problem.

**Why three seeds?**
Three is the number evaluated in the existing experiment, balancing computational cost and seed diversity. There is no study proving it is optimal. The three saved networks must all be present; the local app does not silently use fewer.

**Why 16 kHz?**
It is the experiment's fixed sampling rate and gives an approximately 8-kHz Nyquist frequency while keeping computation modest. Resampling at inference preserves the training input convention. This choice can discard information from higher-rate recordings.

**Why two seconds?**
The experiment uses FoR's two-second subset and fixed 32,000-sample inputs. Short clips are padded and long clips use the first two seconds. The result is not a verdict about every later segment in a long recording.

**What is ROC-AUC?**
ROC-AUC measures how well scores rank positives above negatives across thresholds. It is distinct from accuracy at 0.5. Re-recorded AUC is 0.758 even though fake recall at the deployed threshold is only 11.7%.

**What is EER?**
It is the error level near the point where false-positive and false-negative rates are equal. My code selects the closest discrete ROC point and averages its two error rates. The saved ensemble EERs are 0.118 in-domain and 0.278 re-recorded.

**Why report EER?**
It summarizes the tradeoff between accepting fake audio and rejecting real audio at a balanced operating region. It complements AUC and threshold-specific metrics; it is not the only criterion needed for a deployment decision.

**What is a false positive?**
With synthetic as class 1, it is genuine speech called synthetic. The in-domain matrix contains 12; the re-recorded matrix contains 27. Such errors matter because a detector can wrongly cast doubt on a genuine recording.

**What is a false negative?**
It is synthetic speech called genuine. There are 66 in-domain and 265 re-recorded false negatives, out of 300 synthetic clips in each set. This makes the re-recorded limitation especially important.

**Why does performance drop on re-recorded audio?**
The acoustic channel changes the signal. Frequency response, room reverberation and noise are plausible influences, but the experiment does not isolate their individual causes. The observed drop is evidence of domain sensitivity, not proof of a specific internal mechanism.

**How does noise affect the model?**
The saved Gaussian-noise experiment shows AUC/EER changing from 0.955/0.118 clean to 0.907/0.167 at 20 dB, 0.880/0.205 at 10 dB, and 0.798/0.288 at 0 dB. These are one recorded evaluation, without repeated-draw confidence intervals.

**How does compression affect it?**
The 64-level μ-law encode/decode condition yields AUC 0.871 and EER 0.223. Resampling through 8 kHz yields 0.813 and 0.253. This demonstrates sensitivity to these transformations; it does not establish performance on MP3, AAC or Opus.

**How did you prevent leakage?**
The code preserves the dataset's existing train/validation/test directories and chooses classical-model thresholds using validation sets. But I cannot claim a complete leakage audit: no duplicate hashing, speaker grouping or original-source overlap verification is present. A separately printed test-EER threshold is explicitly optimistic and is not used by the app. Re-recorded validation influenced earlier development, so it was not a wholly unseen domain.

**What does the CNN actually learn?**
It learns filters that improve discrimination on its training distribution. I can describe the feature representation and architecture, but cannot prove it learns only synthesis artifacts. Loudness alone has in-domain test AUC 0.787, showing a possible dataset cue. No causal speaker or saliency study was performed.

**Did you test unseen generators?**
I tested the existing re-recorded domain-shift subset. The provided code does not establish generator identity exclusion. I would not equate re-recording with a demonstrated unseen-generator test; that part of the assignment remains only partially supported.

**What are the main limitations?**
Weak re-recorded recall, sensitivity to degradation, no verified speaker/source-disjoint audit, no dedicated speaker ablation, no calibrated probabilities, and only the first two seconds analyzed. The original checkpoint files are currently missing, so live detection cannot yet be verified.

**Can it detect every AI voice?**
No. The experiment does not cover every generator, language, voice, codec or recording environment. A high in-domain score does not establish universal detection.

**Can a prediction be considered forensic proof?**
No. It is a research model score. It can make both kinds of errors, especially under domain shift, and has not undergone a forensic validation process.

**How would you improve it?**
First recover and fingerprint the original artifacts and audit the data splits. In a future authorized experiment, evaluate held-out generators and speaker/channel ablations, calibrate on validation data, and broaden robustness tests. I did not retrain during this finalization task.

**Which threshold does the demo use, and why?**
0.5, because that is the operating point in the actual cell-12 ensemble evaluation that produced the reported metrics and matrices. A later Gradio cell's 0.871 came from an earlier single-model threshold and was inconsistent with the ensemble evidence.

**Are the weights actually trained?**
No local checkpoint candidates were found, so I cannot currently verify actual files. When recovered, the loader requires three complete compatible finite state dictionaries and positive BatchNorm training counters. This rejects the known random-initialization export, but cannot prove that a later retrained file is the exact original evaluated model. File provenance and inference regression remain necessary.

**Why not just recreate the missing checkpoints?**
Random initialization cannot reproduce trained weights, and new training would be a different experiment. The original notebook contains a cell that writes random networks under the expected names; I preserved it as evidence but never executed it. The app fails explicitly instead of presenting false predictions.

**What is working in the local application today?**
The server, upload/decoder, audio metadata, waveform and Log-Mel preview, API schema and missing-model handling are implemented and tested. Real three-CNN inference and the two labelled speech demos remain blocked by missing model/audio artifacts. Historical metrics are shown as historical results, not current predictions.
