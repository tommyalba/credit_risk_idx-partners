"""FastAPI scoring service for the production bundles.

Exposes the same scoring paths as the CLIs:

    GET  /health                      bundle status
    POST /v1/score/application        Scenario A PD + approve/reject
    POST /v1/score/lifecycle          both scenarios on one record set
    POST /v1/score/monitor            PSI drift verdict for a batch

Request bodies are {"records": [{...raw columns...}, ...]}; a single
record may also be posted as a bare JSON object. Payload validation
(FORBIDDEN_FEATURES, required raw columns) is enforced by the same
`validate_payload` used by the batch CLIs — violations return HTTP 422.

Bundles load lazily on first request and are cached for the process
lifetime. `create_app(bundle_a=..., bundle_b=...)` allows tests to inject
fixtures.

Run:
    uvicorn pipeline.serve:app --host 0.0.0.0 --port 8000
"""
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request

logger = logging.getLogger('credit_risk.serve')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import infer as lifecycle  # noqa: E402
from pipeline.core import add_conformal_intervals, drift_report  # noqa: E402
from pipeline.scenario_a import score as score_a  # noqa: E402
from pipeline.scenario_a import features as feat_a  # noqa: E402
from pipeline.scenario_b import score as score_b  # noqa: E402
from pipeline.scenario_b import features as feat_b  # noqa: E402


def _to_frame(payload) -> pd.DataFrame:
    if isinstance(payload, dict) and 'records' in payload:
        records = payload['records']
    elif isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise HTTPException(422, 'body must be a record object, a list of '
                                 'records, or {"records": [...]}')
    if not records:
        raise HTTPException(422, 'records is empty')
    if not all(isinstance(r, dict) for r in records):
        raise HTTPException(422, 'every record must be a JSON object')
    return pd.DataFrame(records)


def _records_out(df: pd.DataFrame) -> list:
    return df.reset_index(drop=True).to_dict(orient='records')


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def create_app(bundle_a=None, bundle_b=None) -> FastAPI:
    app = FastAPI(title='credit-risk scoring', version='1.0.0')
    bundles = {'a': bundle_a, 'b': bundle_b}

    @app.middleware('http')
    async def access_log(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        logger.info('%s %s -> %s (%.0f ms)',
                    request.method, request.url.path, response.status_code,
                    (time.perf_counter() - t0) * 1000)
        return response

    def get(scenario):
        if bundles[scenario] is None:
            loader = score_a.load_bundle if scenario == 'a' else score_b.load_bundle
            bundles[scenario] = loader()
        return bundles[scenario]

    @app.get('/health')
    def health():
        return {
            'status': 'ok',
            'scenario_a_bundle': str(score_a.BUNDLE_PATH),
            'scenario_b_bundle': str(score_b.BUNDLE_PATH),
            'bundle_a_present': Path(score_a.BUNDLE_PATH).exists(),
            'bundle_b_present': Path(score_b.BUNDLE_PATH).exists(),
        }

    @app.post('/v1/score/application')
    async def score_application(request: Request,
                                intervals: bool = Query(False)):
        df = _to_frame(await request.json())
        bundle = get('a')
        res = _guard(score_a.score, df, bundle=bundle)
        if intervals:
            res = _guard(add_conformal_intervals, res, bundle,
                         feat_a.add_scenario_a_features,
                         Path('data/interim'), score_col='pd_score')
        return {'n': len(res), 'scores': _records_out(res)}

    @app.post('/v1/score/lifecycle')
    async def score_lifecycle_endpoint(request: Request,
                                       intervals: bool = Query(False)):
        df = _to_frame(await request.json())
        result = _guard(lifecycle.score_lifecycle,
                        df, bundle_a=get('a'), bundle_b=get('b'),
                        intervals=intervals)
        return {'n': len(result['combined']),
                'scores': _records_out(result['combined'])}

    @app.post('/v1/score/monitor')
    async def score_monitor(request: Request,
                            scenario: str = Query('a', pattern='^[ab]$')):
        df = _to_frame(await request.json())
        if scenario == 'a':
            rep = _guard(drift_report, df, get('a'),
                         feat_a.add_scenario_a_features,
                         feat_a.FORBIDDEN_FEATURES,
                         score_a.REQUIRED_RAW,
                         'data/interim')
        else:
            rep = _guard(drift_report, df, get('b'),
                         feat_b.add_scenario_b_features,
                         feat_b.FORBIDDEN_FEATURES,
                         score_b.REQUIRED_RAW,
                         'data/interim/scenario_b')
        feats = rep['features']
        top = (feats.head(10).to_dict(orient='records') if len(feats) else [])
        return {
            'scenario': scenario,
            'n_records': len(df),
            'score_psi': rep['score_psi'],
            'score_verdict': rep['score_verdict'],
            'worst_verdict': rep['worst_verdict'],
            'top_features': top,
        }

    return app


app = create_app()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
