"""RunPod Serverless handler.

기존 FastAPI(`main.py`)와 동일한 복원 파이프라인(`restore_image`)을 그대로 재사용하되,
RunPod Serverless 규격(`{"input": {...}}` → `{"output": {...}}`)에 맞춰 감싼 진입점입니다.

Serverless는 요청이 없으면 0으로 스케일 다운되므로(scale-to-zero) 상시 가동 전기 비용 없이
호출당 과금으로 GPU 복원을 돌릴 수 있습니다. 자체 GPU 서버로 전환할 때는 이 파일을 건드릴 필요
없이 클라이언트(modoo-studio)에서 `PHOTO_RESTORE_PROVIDER`만 `http`로 바꾸면 됩니다.

입력(input):
  - image_base64 (str, 필수): 원본 이미지. `data:image/...;base64,` 접두사 허용.
  - mode (str): face|upscale|full|safe|pretty|restore|enhance (기본 enhance)
  - upscale (int): 1|2|4 (기본 2)
  - fidelity (float): 0.0~1.0 (기본 0.7)
  - face_model (str): auto|gfpgan|codeformer (기본 auto)
  - face_weight (float|None): 0.0~1.0 (있으면 fidelity보다 우선)
  - bg_model (str): auto|swinir|light|none (기본 auto)
  - format (str): jpg|jpeg|png|webp (기본 jpg)
  - quality (int): 1~100 (기본 92)

출력(output):
  - image_base64 (str): 복원 결과 이미지(base64, 접두사 없음)
  - format (str)
  - content_type (str)
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
from pathlib import Path

import runpod

from pipelines.restore import restore_image
from pipelines.utils import build_job_paths, ensure_dir, remove_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("photo-restore-handler")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
INPUT_DIR = ensure_dir(os.getenv("INPUT_DIR", "/app/input"))
OUTPUT_DIR = ensure_dir(os.getenv("OUTPUT_DIR", "/app/output"))
MAX_DECODED_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

VALID_MODES = {"face", "upscale", "full", "safe", "pretty", "restore", "enhance"}
VALID_UPSCALES = {1, 2, 4}
VALID_FACE_MODELS = {"auto", "gfpgan", "codeformer"}
VALID_BG_MODELS = {"auto", "swinir", "light", "none"}
OUTPUT_FORMATS = {
    "jpg": (".jpg", "image/jpeg"),
    "jpeg": (".jpg", "image/jpeg"),
    "png": (".png", "image/png"),
    "webp": (".webp", "image/webp"),
}


def _decode_image(image_base64: str) -> bytes:
    if not isinstance(image_base64, str) or not image_base64.strip():
        raise ValueError("image_base64 is required")
    # data URL 접두사 제거
    if "," in image_base64 and image_base64.strip().lower().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if not raw:
        raise ValueError("decoded image is empty")
    if len(raw) > MAX_DECODED_BYTES:
        raise ValueError(f"decoded image exceeds {MAX_DECODED_BYTES} bytes")
    return raw


def _clamp_float(value, default: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        if value is None:
            return default
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def handler(event: dict) -> dict:
    job_input = (event or {}).get("input") or {}

    try:
        raw_bytes = _decode_image(job_input.get("image_base64"))

        mode = str(job_input.get("mode", "enhance")).strip().lower()
        if mode not in VALID_MODES:
            mode = "enhance"

        try:
            upscale = int(job_input.get("upscale", 2))
        except (TypeError, ValueError):
            upscale = 2
        if upscale not in VALID_UPSCALES:
            upscale = 2

        fidelity = _clamp_float(job_input.get("fidelity"), 0.7)
        face_weight_raw = job_input.get("face_weight")
        face_weight = None if face_weight_raw is None else _clamp_float(face_weight_raw, 0.7)

        face_model = str(job_input.get("face_model", "auto")).strip().lower()
        if face_model not in VALID_FACE_MODELS:
            face_model = "auto"

        bg_model = str(job_input.get("bg_model", "auto")).strip().lower()
        if bg_model not in VALID_BG_MODELS:
            bg_model = "auto"

        output_format = str(job_input.get("format", "jpg")).strip().lower()
        if output_format not in OUTPUT_FORMATS:
            output_format = "jpg"

        try:
            quality = int(job_input.get("quality", 92))
        except (TypeError, ValueError):
            quality = 92
        quality = max(1, min(100, quality))

        output_suffix, media_type = OUTPUT_FORMATS[output_format]
        input_path, output_path, _ = build_job_paths(
            INPUT_DIR,
            OUTPUT_DIR,
            "input.png",
            output_suffix=output_suffix,
        )
        input_path.write_bytes(raw_bytes)

        try:
            result_path = Path(
                restore_image(
                    input_path,
                    output_path,
                    mode,
                    upscale,
                    fidelity,
                    quality,
                    MODEL_DIR,
                    face_model,
                    face_weight,
                    bg_model,
                )
            )
            if not result_path.exists():
                raise RuntimeError("restored file was not created")

            encoded = base64.b64encode(result_path.read_bytes()).decode("ascii")
        finally:
            remove_files([input_path, output_path])

        return {
            "image_base64": encoded,
            "format": "jpg" if output_format == "jpeg" else output_format,
            "content_type": media_type,
        }
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - serverless 핸들러는 항상 dict를 돌려줘야 함
        logger.exception("restore handler failed")
        return {"error": str(exc)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
