from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from gfpgan import GFPGANer

from .upscale import UpscalePipeline, load_image, save_image
from .utils import ensure_dir

VALID_MODES = {"face", "upscale", "full", "safe", "pretty", "restore", "enhance"}


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

    def enhance_background(self, image_bgr: np.ndarray, strength: float) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return image_bgr

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        clip_limit = 1.2 + (1.8 * strength)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)
        contrast = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)

        denoise_h = 2 + int(5 * strength)
        denoised = cv2.fastNlMeansDenoisingColored(
            contrast,
            None,
            denoise_h,
            denoise_h,
            7,
            21,
        )

        blur = cv2.GaussianBlur(denoised, (0, 0), 1.2)
        sharpen_amount = 0.25 + (0.45 * strength)
        sharpened = cv2.addWeighted(
            denoised,
            1.0 + sharpen_amount,
            blur,
            -sharpen_amount,
            0,
        )
        return sharpened

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        mode: str,
        upscale: int,
        fidelity: float,
        output_quality: int,
    ) -> Path:
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        if upscale not in (1, 2, 4):
            raise ValueError("upscale must be one of 1, 2, 4")
        if not 0.0 <= fidelity <= 1.0:
            raise ValueError("fidelity must be between 0.0 and 1.0")
        output_quality = int(np.clip(output_quality, 1, 100))

        image = load_image(input_path)
        result = image.copy()

        if mode in {"full", "restore"}:
            result = self.enhance_background(result, 0.35)
        elif mode == "safe":
            result = self.enhance_background(result, 0.2)
        elif mode in {"pretty", "enhance"}:
            result = self.enhance_background(result, 0.65)

        if mode in {"face", "full", "safe", "pretty", "restore", "enhance"}:
            applied_fidelity = fidelity
            if mode == "safe":
                applied_fidelity = min(fidelity, 0.5)
            elif mode in {"pretty", "enhance"}:
                applied_fidelity = max(fidelity, 0.85)
            result = self.restore_faces(result, applied_fidelity)

        if mode in {"upscale", "full", "safe", "pretty", "restore", "enhance"} and upscale > 1:
            result = self._upscale_pipeline.upscale(result, upscale)

        output_path = Path(output_path)
        save_image(output_path, result, quality=output_quality)
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
    output_quality: int,
    model_dir: str | Path = "/app/models",
) -> Path:
    pipeline = get_pipeline(model_dir=model_dir)
    return pipeline.process(
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        upscale=upscale,
        fidelity=fidelity,
        output_quality=output_quality,
    )
