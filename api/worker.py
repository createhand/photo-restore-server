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

    def submit(self, input_path: str | Path, output_path: str | Path, mode: str, upscale: int, fidelity: float) -> Future:
        return self.executor.submit(
            restore_image,
            str(input_path),
            str(output_path),
            mode,
            upscale,
            fidelity,
            MODEL_DIR,
        )


worker = RestoreWorker(max_workers=MAX_GPU_WORKERS)
