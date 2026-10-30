# Original research archive — do not Run All

`original.ipynb` and `original_export.py.txt` are byte-preserved copies of the user-supplied notebook and exported Python file. The export has a `.txt` suffix to prevent accidental execution/import. [Source hashes](../reports/source_manifest.json) record their integrity. `assignment.jpg` is the supplied assignment specification.

This archive is evidence, not the deployment entrypoint. It includes environment-specific paths, training loops, inconsistent later reporting, a cell that creates random replacement checkpoints, and a later retraining. None were executed during finalization. The local app imports only `app/` and `src/`; it does not require Colab, Kaggle or Gradio.

Zero-based notebook cell map:

| Cell | Evidence |
|---|---|
| 2 | Dataset inventory and sample format |
| 9 | 1,500/300/300 per-class subset, preprocessing, single-CNN experiment |
| 10 | RF Log-Mel/MFCC baselines, RMS/silence check, single-CNN metrics |
| 11 | Twelve-epoch CNN seeds 0, 1, 2 and individual AUC/EER outputs |
| 12 | Evaluated three-CNN ensemble at 0.5, matrices and robustness |
| 14 | Later Gradio predictor incorrectly using the earlier 0.871 threshold |
| 19 | Dangerous random-weight export under the expected checkpoint names |
| 20 | Later prose incorrectly linking ensemble results to 0.871 |
| 29–30 | Remote checkpoint search; only the Mel transform recovered |
| 32–33 | Later five-epoch training with same filenames; not the evaluated ensemble |

Saved notebook outputs can originate from different execution sessions. Execution-order consistency is not guaranteed. The local report prioritizes explicit evaluation code and matching saved output, and labels provenance gaps instead of treating every later prose claim as verified.
