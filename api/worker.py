from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from pipelines.restore import restore_image

MAX_GPU_WORKERS = max(1, int(os.getenv("MAX_GPU_WORKERS", "1")))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))


class RestoreWorker:
    def __init__(self, max_workers: int = MAX_GPU_WORKERS) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gpu-worker")

    def submit(
        self,
        input_path: str | Path,
        output_path: str | Path,
        mode: str,
        upscale: int,
        fidelity: float,
        output_quality: int,
        face_model: str = "auto",
        face_weight: float | None = None,
        bg_model: str = "auto",
    ) -> Future:
        return self.executor.submit(
            restore_image,
            str(input_path),
            str(output_path),
            mode,
            upscale,
            fidelity,
            output_quality,
            MODEL_DIR,
            face_model,
            face_weight,
            bg_model,
        )


worker = RestoreWorker(max_workers=MAX_GPU_WORKERS)
