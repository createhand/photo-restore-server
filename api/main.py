from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from pipelines.utils import build_job_paths, ensure_dir, remove_files, validate_image_extension
from worker import worker

APP_NAME = "photo-restore-server"
logger = logging.getLogger(APP_NAME)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
INPUT_DIR = ensure_dir(os.getenv("INPUT_DIR", "/app/input"))
OUTPUT_DIR = ensure_dir(os.getenv("OUTPUT_DIR", "/app/output"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
VALID_MODES = {"face", "upscale", "full", "safe", "pretty", "restore", "enhance"}
VALID_UPSCALES = {1, 2, 4}
OUTPUT_FORMATS = {
    "jpg": (".jpg", "image/jpeg"),
    "jpeg": (".jpg", "image/jpeg"),
    "png": (".png", "image/png"),
    "webp": (".webp", "image/webp"),
}
ALLOWED_CONTENT_TYPES = {
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/webp",
}
REQUIRED_MODELS = (
    "GFPGANv1.4.pth",
    "RealESRGAN_x2.pth",
    "RealESRGAN_x4.pth",
)
OPTIONAL_MODELS = {
    "codeformer": ("CodeFormer.pth", "codeformer.pth"),
    "swinir": ("SwinIR-M_x4_GAN.pth", "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-M_x4_GAN.pth"),
}
VALID_FACE_MODELS = {"auto", "gfpgan", "codeformer"}
VALID_BG_MODELS = {"auto", "swinir", "light", "none"}

app = FastAPI(title=APP_NAME, version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": APP_NAME}


@app.get("/ready")
def ready():
    missing_models = [name for name in REQUIRED_MODELS if not (MODEL_DIR / name).exists()]
    optional_models = {
        key: any((MODEL_DIR / name).exists() for name in names)
        for key, names in OPTIONAL_MODELS.items()
    }
    status_code = 200 if not missing_models else 503
    content = {
        "status": "ok" if not missing_models else "degraded",
        "service": APP_NAME,
        "models_ready": not missing_models,
        "missing_models": missing_models,
        "optional_models": optional_models,
    }
    return JSONResponse(status_code=status_code, content=content)


async def read_upload(file: UploadFile) -> bytes:
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported content type: {file.content_type}",
        )

    try:
        validate_image_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"uploaded file exceeds {MAX_UPLOAD_BYTES} bytes",
        )
    return contents


@app.post("/restore")
async def restore_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("full"),
    upscale: int = Form(2),
    fidelity: float = Form(0.7),
    face_model: str = Form("auto"),
    face_weight: float | None = Form(None),
    bg_model: str = Form("auto"),
    format: str = Form("jpg"),
    output_format: str | None = Form(None),
    quality: int = Form(90),
):
    output_format = (output_format or format).lower()
    face_model = face_model.lower()
    bg_model = bg_model.lower()
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {sorted(VALID_MODES)}")
    if upscale not in VALID_UPSCALES:
        raise HTTPException(status_code=400, detail="upscale must be one of 1, 2, 4")
    if not 0.0 <= fidelity <= 1.0:
        raise HTTPException(status_code=400, detail="fidelity must be between 0.0 and 1.0")
    if face_model not in VALID_FACE_MODELS:
        raise HTTPException(status_code=400, detail=f"face_model must be one of {sorted(VALID_FACE_MODELS)}")
    if face_weight is not None and not 0.0 <= face_weight <= 1.0:
        raise HTTPException(status_code=400, detail="face_weight must be between 0.0 and 1.0")
    if bg_model not in VALID_BG_MODELS:
        raise HTTPException(status_code=400, detail=f"bg_model must be one of {sorted(VALID_BG_MODELS)}")
    if output_format not in OUTPUT_FORMATS:
        raise HTTPException(status_code=400, detail=f"format must be one of {sorted(OUTPUT_FORMATS)}")
    if not 1 <= quality <= 100:
        raise HTTPException(status_code=400, detail="quality must be between 1 and 100")

    try:
        output_suffix, media_type = OUTPUT_FORMATS[output_format]
        input_path, output_path, _ = build_job_paths(
            INPUT_DIR,
            OUTPUT_DIR,
            file.filename,
            output_suffix=output_suffix,
        )
        contents = await read_upload(file)
        input_path.write_bytes(contents)

        future = worker.submit(
            input_path,
            output_path,
            mode,
            upscale,
            fidelity,
            quality,
            face_model,
            face_weight,
            bg_model,
        )
        result_path = Path(await run_in_threadpool(future.result))

        if not result_path.exists():
            raise HTTPException(status_code=500, detail="restored file was not created")

        return FileResponse(
            path=result_path,
            media_type=media_type,
            filename=result_path.name,
            background=BackgroundTask(remove_files, [input_path]),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("restore failed")
        remove_files(
            [
                path
                for path in (
                    locals().get("input_path"),
                    locals().get("output_path"),
                )
                if path is not None
            ]
        )
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.on_event("startup")
def startup_event() -> None:
    ensure_dir(MODEL_DIR)
    ensure_dir(INPUT_DIR)
    ensure_dir(OUTPUT_DIR)
