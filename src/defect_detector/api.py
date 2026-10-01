"""FastAPI inference service."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .exceptions import InvalidImageError, ModelArtifactError
from .inference import Prediction, Predictor
from .logging_config import configure_logging

LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
_UNSET = object()


class PredictionResponse(BaseModel):
    request_id: str
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model_version: str


def create_app(
    model_path: Path | None | object = _UNSET, predictor: Predictor | None = None
) -> FastAPI:
    """Create an app; dependency injection keeps API tests independent of model files."""
    if model_path is _UNSET:
        configured_path: Path | None = Path(os.getenv("DEFECT_MODEL_PATH", "models/best.pt"))
    elif model_path is None:
        configured_path = None
    else:
        if not isinstance(model_path, Path):
            raise TypeError("model_path must be a pathlib.Path or None")
        configured_path = model_path

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        if predictor is not None:
            application.state.predictor = predictor
        elif configured_path is not None and configured_path.is_file():
            try:
                application.state.predictor = Predictor(configured_path)
            except ModelArtifactError:
                LOGGER.exception("Model artifact failed to load: %s", configured_path)
                application.state.predictor = None
        elif configured_path is None:
            LOGGER.info("Model loading disabled for this app instance")
            application.state.predictor = None
        else:
            LOGGER.warning(
                "Model artifact not found; readiness and prediction will be unavailable: %s",
                configured_path,
            )
            application.state.predictor = None
        yield

    application = FastAPI(
        title="Visual Defect Detector",
        version="0.1.0",
        description="Image classification API for normal and defective products.",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.get("/health")
    async def health(request: Request) -> dict[str, object]:
        return {"status": "ok", "model_loaded": request.app.state.predictor is not None}

    @application.get("/ready")
    async def ready(request: Request) -> dict[str, object]:
        if request.app.state.predictor is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return {"status": "ready"}

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image content type")
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 10 MiB limit")
        service_predictor: Predictor | None = request.app.state.predictor
        if service_predictor is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        try:
            prediction: Prediction = service_predictor.predict_bytes(content)
        except InvalidImageError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        LOGGER.info(
            "prediction request_id=%s class=%s confidence=%.4f filename=%s",
            request.state.request_id,
            prediction.predicted_class,
            prediction.confidence,
            file.filename,
        )
        return PredictionResponse(request_id=request.state.request_id, **prediction.__dict__)

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception(
            "Unhandled request error request_id=%s", getattr(request.state, "request_id", "unknown")
        )
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return application


app = create_app()


def run() -> None:
    """Console entry point."""
    import uvicorn

    uvicorn.run("defect_detector.api:app", host="0.0.0.0", port=8000)
