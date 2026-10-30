from contextlib import asynccontextmanager
from pathlib import Path
import os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from src.audio import FeatureExtractor, decode_audio, MAX_BYTES
from src.model import Ensemble

ROOT = Path(__file__).resolve().parents[1]


def create_app(model_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        app.state.extractor = FeatureExtractor()
        app.state.ensemble = Ensemble(model_dir or os.environ.get("DEEPGUARD_MODEL_DIR", ROOT / "models"))
        yield

    app = FastAPI(title="DeepGuard", version="1.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "app" / "static" / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#142f31"/><path d="M16 6 26 16 16 26 6 16Z" fill="none" stroke="#bcebd5" stroke-width="2"/></svg>', media_type="image/svg+xml")

    @app.get("/api/health")
    def health():
        ensemble = app.state.ensemble
        return {"status": "ok" if ensemble.ready else "degraded", "model_loaded": ensemble.ready,
                "models": len(ensemble.models), "missing_models": ensemble.missing,
                "detail": ensemble.error, "threshold": 0.5, "checkpoint_sha256": ensemble.hashes,
                "checkpoint_provenance": "not independently matched to historical metrics"}

    async def read_upload(file):
        try:
            data = await file.read(MAX_BYTES + 1)
        finally:
            await file.close()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Audio upload exceeds 20 MB.")
        try:
            return await run_in_threadpool(decode_audio, data, file.filename or "audio", app.state.extractor)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/features")
    async def features(file: UploadFile = File(...)):
        metadata, _ = await read_upload(file)
        return metadata

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...)):
        if not app.state.ensemble.ready:
            await file.close()
            raise HTTPException(503, app.state.ensemble.error)
        metadata, features = await read_upload(file)
        try:
            result = await run_in_threadpool(app.state.ensemble.predict, features)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(500, "Model inference failed; no prediction was produced.") from exc
        return {**metadata, **result}

    return app


app = create_app()
