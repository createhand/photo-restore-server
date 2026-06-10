from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from gfpgan import GFPGANer

from .upscale import UpscalePipeline, load_image, save_image
from .utils import ensure_dir

VALID_MODES = {"face", "upscale", "full", "safe", "pretty"}


class RestorePipeline:
    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        self.model_dir = ensure_dir(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._face_restorer: GFPGANer | None = None
        self._upscale_pipeline = UpscalePipeline(model_dir=self.model_dir, device=self.device)

    def _get_face_restorer(self) -> GFPGANer:
        if self._face_restorer is None:
            model_path = self.model_dir / "GFPGANv1.4.pth"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Missing GFPGAN model file: {model_path}. Place the weight file in /app/models"
                )

            self._face_restorer = GFPGANer(
                model_path=str(model_path),
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device,
            )
        return self._face_restorer

    def restore_faces(self, image_bgr: np.ndarray, fidelity: float) -> np.ndarray:
        restorer = self._get_face_restorer()
        _, _, restored = restorer.enhance(
            image_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=fidelity,
        )
        if restored is None:
            raise ValueError("GFPGAN did not return a restored image")
        return restored

    def process(self, input_path: str | Path, output_path: str | Path, mode: str, upscale: int, fidelity: float) -> Path:
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        if upscale not in (1, 2, 4):
            raise ValueError("upscale must be one of 1, 2, 4")
        if not 0.0 <= fidelity <= 1.0:
            raise ValueError("fidelity must be between 0.0 and 1.0")

        image = load_image(input_path)
        result = image.copy()

        if mode in {"face", "full", "safe", "pretty"}:
            applied_fidelity = fidelity
            if mode == "safe":
                applied_fidelity = min(fidelity, 0.5)
            elif mode == "pretty":
                applied_fidelity = max(fidelity, 0.85)
            result = self.restore_faces(result, applied_fidelity)

        if mode in {"upscale", "full", "safe", "pretty"} and upscale > 1:
            result = self._upscale_pipeline.upscale(result, upscale)

        output_path = Path(output_path)
        save_image(output_path, result)
        return output_path


_pipeline: RestorePipeline | None = None


def get_pipeline(model_dir: str | Path = "/app/models") -> RestorePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RestorePipeline(model_dir=model_dir)
    return _pipeline


def restore_image(
    input_path: str | Path,
    output_path: str | Path,
    mode: str,
    upscale: int,
    fidelity: float,
    model_dir: str | Path = "/app/models",
) -> Path:
    pipeline = get_pipeline(model_dir=model_dir)
    return pipeline.process(
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        upscale=upscale,
        fidelity=fidelity,
    )
