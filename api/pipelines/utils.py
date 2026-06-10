from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Iterable

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sanitize_filename(filename: str) -> str:
    original = Path(filename).name
    stem = Path(original).stem
    suffix = Path(original).suffix.lower()
    safe_stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._") or "image"
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".png"
    return f"{safe_stem}{suffix}"


def build_job_paths(input_dir: str | Path, output_dir: str | Path, filename: str) -> tuple[Path, Path, str]:
    safe_name = sanitize_filename(filename)
    job_id = uuid.uuid4().hex
    input_path = ensure_dir(input_dir) / f"{job_id}_{safe_name}"
    output_path = ensure_dir(output_dir) / f"{job_id}_restored.png"
    return input_path, output_path, job_id


def validate_image_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension: {suffix or 'none'}. Allowed: {allowed}")


def remove_files(paths: Iterable[str | Path]) -> None:
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            continue
