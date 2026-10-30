from pathlib import Path
from io import BytesIO
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from src.evaluate import metrics
from src.audio import FeatureExtractor, decode_audio, prepare_waveform
import torch

ROOT=Path(__file__).resolve().parents[1]


def test_opus_upload_matches_research_soundfile_read():
    path=ROOT/'demo/audio_2026-10-30_18-28-14.ogg'
    audio,sr=sf.read(path,dtype='float32')
    extractor=FeatureExtractor()
    expected=extractor(prepare_waveform(audio,sr))[1]
    _,actual=decode_audio(path.read_bytes(),path.name,extractor)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_supplied_models_and_genuine_upload():
    with TestClient(create_app(ROOT/'models')) as client:
        health=client.get('/api/health').json()
        assert health['models']==3 and health['model_loaded']
        assert len(set(health['checkpoint_sha256'].values()))==3
        path=ROOT/'demo/audio_2026-10-30_18-28-14.ogg'
        with path.open('rb') as f:
            response=client.post('/api/predict',files={'file':(path.name,f,'audio/ogg')})
        assert response.status_code==200,response.text
        data=response.json()
        assert len(data['individual_predictions'])==3
        assert data['fake_probability']==pytest.approx(sum(data['seed_probabilities'])/3)
        assert data['confidence']==max(data['fake_probability'],data['genuine_probability'])
        # This asserts a working pipeline, not correct classification of every input.
        assert client.post('/api/predict',files={'file':('bad.wav',b'bad')}).status_code==422


def test_single_class_metrics_not_fabricated():
    result=metrics([0],[.2])
    assert result['accuracy']==1 and result['confusion_matrix']==[[1,0],[0,0]]
    assert result['roc_auc'] is None and result['eer'] is None and result['recall'] is None


def test_two_class_metrics_known_confusion_counts():
    result=metrics([0,0,1,1],[.1,.6,.7,.8])
    assert result['confusion_matrix']==[[1,1],[0,2]]
    assert result['accuracy']==.75 and result['precision']==pytest.approx(2/3)
    assert result['recall']==1 and result['f1']==pytest.approx(.8)
    assert result['roc_auc']==1 and result['eer']==0
