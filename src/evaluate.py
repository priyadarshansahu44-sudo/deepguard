"""Inference-only evaluation from explicitly labelled manifests; never trains."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from src.audio import prepare_waveform, FeatureExtractor
from src.model import Ensemble


def metrics(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities)
    predicted = (p >= .5).astype(int)
    result = {"n": len(y), "accuracy": float(accuracy_score(y, predicted)),
              "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
              "precision": None, "recall": None, "f1": None, "roc_auc": None, "eer": None}
    if len(set(y)) != 2:
        result["limitation"] = "Only one ground-truth class available. Discrimination metrics are not reported; accuracy is a single-class smoke check."
        return result
    fpr, tpr, _ = roc_curve(y, p)
    i = np.nanargmin(np.abs((1-tpr)-fpr))
    result.update(precision=float(precision_score(y,predicted,zero_division=0)),
                  recall=float(recall_score(y,predicted,zero_division=0)),
                  f1=float(f1_score(y,predicted,zero_division=0)),
                  roc_auc=float(roc_auc_score(y,p)), eer=float((fpr[i]+1-tpr[i])/2))
    return result


def evaluate(manifest, model_dir, output):
    manifest = Path(manifest).resolve()
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("Manifest must contain labelled samples.")
    torch.set_num_threads(4)
    ensemble, extractor = Ensemble(model_dir), FeatureExtractor()
    if not ensemble.ready:
        raise ValueError(ensemble.error)
    generator = torch.Generator().manual_seed(20261030)
    rows, seen = [], set()
    for record in records:
        if record.get("label") not in (0,1) or not record.get("label_source"):
            raise ValueError("Every sample requires label 0/1 and an explicit label_source.")
        path = (manifest.parent / record['path']).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError("Duplicate sample bytes in evaluation manifest.")
        seen.add(digest)
        audio, sr = sf.read(path, dtype='float32')
        x = prepare_waveform(audio, sr)
        variants = {'clean':x}
        for snr in [20,10,0]:
            variants[f'noise_{snr}dB'] = x + torch.randn(x.shape,generator=generator)*torch.sqrt(x.pow(2).mean(1,keepdim=True)/10**(snr/10))
        variants['mulaw_64'] = torchaudio.functional.mu_law_decoding(torchaudio.functional.mu_law_encoding(x.clamp(-1,1),64),64)
        variants['8khz_roundtrip'] = torchaudio.functional.resample(torchaudio.functional.resample(x,16000,8000),8000,16000)
        predictions = {name: ensemble.predict(extractor(wave)[1]) for name,wave in variants.items()}
        rows.append({**record, 'sha256':digest, 'predictions':predictions})
    conditions = {name:metrics([r['label'] for r in rows],[r['predictions'][name]['fake_probability'] for r in rows]) for name in rows[0]['predictions']}
    unseen = [r for r in rows if r.get('unseen_source_verified') is True and r.get('unseen_source_evidence')]
    result = {'kind':'fresh inference with supplied checkpoints', 'threshold':.5,
              'checkpoint_sha256':ensemble.hashes, 'conditions':conditions, 'samples':rows,
              'unseen_source': metrics([r['label'] for r in unseen],[r['predictions']['clean']['fake_probability'] for r in unseen]) if unseen else {'status':'Not evaluated: no verified source-held-out samples'},
              'note':'Different degradations of a clip are not independent test samples. Historical metrics are not attributed to these checkpoints.'}
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--models',default='models')
    parser.add_argument('--output',default='reports/fresh_evaluation.json')
    args=parser.parse_args()
    result=evaluate(args.manifest,args.models,args.output)
    print(json.dumps(result['conditions'],indent=2))
