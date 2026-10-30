# Saved notebook output evidence

Cell numbers below are zero-based. These are historical outputs, not fresh experiments.

## Cell 2

```text
split       label
testing     0         544
            1         544
training    0        6978
            1        6978
validation  0        1413
            1        1413
dtype: int64
path
wav    17870
Name: count, dtype: int64
sample rate: 16000 | samples: 32000 | seconds: 2.0
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-rerec -> 0 files
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-rerec/for-rerecorded -> 0 files
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-rerec/for-rerecorded/testing -> 0 files
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-rerec/for-rerecorded/testing/fake -> 408 files
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-rerec/for-rerecorded/testing/real -> 408 files
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-rerec/for-rerecorded/training -> 0 files
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-rerec/for-rerecorded/training/fake -> 5104 files
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-rerec/for-rerecorded/training/real -> 5104 files
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-rerec/for-rerecorded/validation -> 0 files
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-rerec/for-rerecorded/validation/fake -> 1143 files
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-rerec/for-rerecorded/validation/real -> 1101 files

```

## Cell 9

```text
train torch.Size([3000, 32000]) 4s
val torch.Size([600, 32000])
test torch.Size([600, 32000])
rerec val torch.Size([600, 32000])
rerec test torch.Size([600, 32000]) 189s
ep1: val(same-src) acc 0.617 EER 0.093 | val(rerec) acc 0.538 AUC 0.800 EER 0.262
ep2: val(same-src) acc 0.958 EER 0.040 | val(rerec) acc 0.530 AUC 0.719 EER 0.320
ep3: val(same-src) acc 0.955 EER 0.027 | val(rerec) acc 0.608 AUC 0.707 EER 0.317
ep4: val(same-src) acc 0.963 EER 0.017 | val(rerec) acc 0.692 AUC 0.772 EER 0.263
ep5: val(same-src) acc 0.955 EER 0.010 | val(rerec) acc 0.738 AUC 0.766 EER 0.278
ep6: val(same-src) acc 0.985 EER 0.015 | val(rerec) acc 0.570 AUC 0.734 EER 0.300
ep7: val(same-src) acc 0.982 EER 0.013 | val(rerec) acc 0.485 AUC 0.730 EER 0.317
ep8: val(same-src) acc 0.990 EER 0.013 | val(rerec) acc 0.497 AUC 0.752 EER 0.293
ep9: val(same-src) acc 0.990 EER 0.010 | val(rerec) acc 0.505 AUC 0.742 EER 0.300
ep10: val(same-src) acc 0.990 EER 0.010 | val(rerec) acc 0.522 AUC 0.745 EER 0.300
ep11: val(same-src) acc 0.995 EER 0.007 | val(rerec) acc 0.518 AUC 0.757 EER 0.292
ep12: val(same-src) acc 0.992 EER 0.010 | val(rerec) acc 0.523 AUC 0.752 EER 0.293
for-2sec test: acc 0.5000 AUC 0.7375 EER 0.3333
for-rerec test: acc 0.5000 AUC 0.8126 EER 0.2517
total 202s

```

## Cell 10

```text
thresholds: for-2sec 0.871 | rerec 0.759

=== CNN (EER threshold from validation) ===
for-2sec test            acc=0.590 P=0.552 R=0.957 F1=0.700 AUC=0.737 EER=0.333
[[ 67 233]
 [ 13 287]]  (rows=true real/fake, cols=pred real/fake)
  false positives (real called fake): 233 | false negatives (fake called real): 13
for-rerec test           acc=0.655 P=0.599 R=0.940 F1=0.732 AUC=0.813 EER=0.252
[[111 189]
 [ 18 282]]  (rows=true real/fake, cols=pred real/fake)
  false positives (real called fake): 189 | false negatives (fake called real): 18

=== Random Forest on log-mel stats ===
for-2sec test            acc=0.682 P=0.950 R=0.383 F1=0.546 AUC=0.923 EER=0.160
for-rerec test           acc=0.583 P=0.588 R=0.557 F1=0.572 AUC=0.607 EER=0.418

=== Random Forest on MFCC stats ===
for-2sec test            acc=0.598 P=0.650 R=0.427 F1=0.515 AUC=0.593 EER=0.433
for-rerec test           acc=0.430 P=0.306 R=0.110 F1=0.162 AUC=0.337 EER=0.602

=== Shortcut check (RMS level, silence fraction) ===
train          RMS real/fake 0.1533/0.1785 | zero-frac real/fake 0.0067/0.0292 | AUC of RMS alone 0.688
for-2sec test  RMS real/fake 0.1688/0.2110 | zero-frac real/fake 0.0103/0.0221 | AUC of RMS alone 0.787
rerec test     RMS real/fake 0.3390/0.4095 | zero-frac real/fake 0.0178/0.0147 | AUC of RMS alone 0.740

=== Robustness (CNN, for-2sec test, fixed threshold) ===
clean                    acc=0.590 P=0.552 R=0.957 F1=0.700 AUC=0.737 EER=0.333
noise 20 dB              acc=0.568 P=0.539 R=0.940 F1=0.685 AUC=0.611 EER=0.415
noise 10 dB              acc=0.515 P=0.508 R=0.997 F1=0.673 AUC=0.646 EER=0.405
noise 0 dB               acc=0.560 P=0.534 R=0.940 F1=0.681 AUC=0.607 EER=0.440
8 kHz bandlimit          acc=0.487 P=0.455 R=0.133 F1=0.206 AUC=0.608 EER=0.433
mu-law 64 levels         acc=0.580 P=0.547 R=0.927 F1=0.688 AUC=0.681 EER=0.367
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-2sec/for-2seconds/testing/real/file10.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav
  P(fake) = 0.946 -> SYNTHETIC
/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-2sec/for-2seconds/testing/fake/file100.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav
  P(fake) = 0.937 -> SYNTHETIC

```

## Cell 11

```text
seed 0: val AUC 0.999 | 2sec test AUC 0.927 EER 0.147 | rerec test AUC 0.678 EER 0.363
seed 1: val AUC 0.999 | 2sec test AUC 0.930 EER 0.152 | rerec test AUC 0.752 EER 0.295
seed 2: val AUC 1.000 | 2sec test AUC 0.948 EER 0.137 | rerec test AUC 0.776 EER 0.258

```

## Cell 12

```text
=== Ensemble of 3 CNNs ===
for-2sec test [thr 0.5]  acc=0.870 P=0.951 R=0.780 F1=0.857 AUC=0.955 EER=0.118
[[288  12]
 [ 66 234]] (rows=true real/fake, cols=pred real/fake)
for-2sec test [thr at test EER (optimistic)] acc=0.882 P=0.883 R=0.880 F1=0.881 AUC=0.955 EER=0.118
[[265  35]
 [ 36 264]] (rows=true real/fake, cols=pred real/fake)
for-rerec test [thr 0.5] acc=0.513 P=0.565 R=0.117 F1=0.193 AUC=0.758 EER=0.278
[[273  27]
 [265  35]] (rows=true real/fake, cols=pred real/fake)
for-rerec test [thr at test EER (optimistic)] acc=0.722 P=0.721 R=0.723 F1=0.722 AUC=0.758 EER=0.278
[[216  84]
 [ 83 217]] (rows=true real/fake, cols=pred real/fake)

=== Robustness (ensemble, for-2sec) ===
clean              AUC 0.955  EER 0.118
noise 20 dB        AUC 0.907  EER 0.167
noise 10 dB        AUC 0.880  EER 0.205
noise 0 dB         AUC 0.798  EER 0.288
8 kHz bandlimit    AUC 0.813  EER 0.253
mu-law 64 levels   AUC 0.871  EER 0.223
P(fake) = 0.033 -> GENUINE
P(fake) = 0.659 -> SYNTHETIC

```

## Cell 19

```text
Re-generated all deliverables: ['.config', 'cnn_seed0.pt', 'cnn_seed2.pt', 'requirements.txt', 'README.md', 'technical_report.md', 'cnn_seed1.pt', 'sample_data']

```

## Cell 29

```text
MODEL WEIGHT FILE SEARCH RESULTS
NO MODEL WEIGHT FILES FOUND

```

## Cell 30

```text
Mounted at /content/drive

--- SEARCHING GOOGLE DRIVE ---
No saved weight files found in Google Drive.

--- CHECKING CURRENT RUNTIME MEMORY ---
MODEL FOUND IN MEMORY: mel
RECOVERED CHECKPOINT: /content/mel_recovered.pt

```

## Cell 32

```text
⌛ Downloading dataset from Kaggle via kagglehub...
Downloading to /root/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/2.archive...
100%|██████████| 16.0G/16.0G [07:13<00:00, 39.7MB/s]Extracting files...

✅ Dataset successfully downloaded to: /root/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2
⌛ Loading light subset of the dataset...
🚀 Training the 3 CNN Ensemble models (Fast-mode)... 
✅ Saved weight: /content/cnn_seed0.pt
✅ Saved weight: /content/cnn_seed1.pt
✅ Saved weight: /content/cnn_seed2.pt

⭐ SUCCESS: Weight files recreated. You can now execute the Gradio Interface cell (fdb1ef2f)!

```

## Cell 33

```text
=== VERIFYING GENERATED MODEL WEIGHTS ===
✅ FOUND: /content/cnn_seed0.pt (0.38 MB)
✅ FOUND: /content/cnn_seed1.pt (0.38 MB)
✅ FOUND: /content/cnn_seed2.pt (0.38 MB)

⭐ SUCCESS: All 3 seed weight files are verified and ready for the ensemble prediction interface!

```

