"""FastAPI serving tests: health, scoring, payload validation."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.serve import create_app  # noqa: E402
from tests.test_scenario_a import _raw_row  # noqa: E402


def _toy_bundle():
    X = pd.DataFrame({'f_a': [0.0, 1.0], 'f_b': [0.2, 0.9]})
    model = DummyClassifier(strategy='prior').fit(X, [0, 1])
    return {
        'model': model, 'calibrator': None, 'calibrator_name': None,
        'features': ['f_a', 'f_b'], 'cat_columns': [],
        'categories': {}, 'threshold': 0.5,
    }


@pytest.fixture()
def client():
    bundle = _toy_bundle()
    app = create_app(bundle_a=bundle, bundle_b=bundle)
    return TestClient(app)


def _payload(**over):
    row = _raw_row(**over)
    row['f_a'] = 0.5
    row['f_b'] = 0.5
    return row


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_score_application_single_record(client):
    r = client.post('/v1/score/application', json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert body['n'] == 1
    s = body['scores'][0]
    assert 0.0 <= s['pd_score'] <= 1.0
    assert s['decision'] in ('APPROVE', 'REJECT')


def test_score_application_batch(client):
    r = client.post('/v1/score/application',
                    json={'records': [_payload(), _payload(dti=25.0)]})
    assert r.status_code == 200
    assert r.json()['n'] == 2


def test_forbidden_column_rejected(client):
    r = client.post('/v1/score/application',
                    json=_payload(recoveries=100.0))
    assert r.status_code == 422
    assert 'leakage' in r.json()['detail'].lower() or 'not allowed' in r.json()['detail']


def test_missing_required_column_rejected(client):
    row = _payload()
    del row['dti']
    r = client.post('/v1/score/application', json=row)
    assert r.status_code == 422
    assert 'missing' in r.json()['detail'].lower()


def test_lifecycle_requires_funding_fields(client):
    # application-only payload cannot be scored by Scenario B
    r = client.post('/v1/score/lifecycle', json=_payload())
    assert r.status_code == 422
