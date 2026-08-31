from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from gfpgan import GFPGANer

from .codeformer import CodeFormerRestorer
from .sizing import resolve_effective_upscale
from .swinir import SwinIRPipeline
from .upscale import UpscalePipeline, load_image, save_image
from .utils import ensure_dir

logger = logging.getLogger("photo-restore-server")

VALID_MODES = {"face", "upscale", "full", "safe", "pretty", "restore", "enhance"}
VALID_FACE_MODELS = {"auto", "gfpgan", "codeformer"}
VALID_BG_MODELS = {"auto", "swinir", "light", "none"}

# Modes that run the strong "pretty" preset.
STRONG_MODES = {"pretty", "enhance"}
# Modes that touch the whole image, not just faces.
BACKGROUND_MODES = {"full", "safe", "pretty", "restore", "enhance"}
FACE_MODES = {"face", "full", "safe", "pretty", "restore", "enhance"}

MAX_INPUT_LONG_SIDE = int(os.getenv("MAX_INPUT_LONG_SIDE", "3600"))
MAX_OUTPUT_PIXELS = int(os.getenv("MAX_OUTPUT_PIXELS", str(40_000_000)))
AUTO_2X_MAX_INPUT_PIXELS = int(os.getenv("AUTO_2X_MAX_INPUT_PIXELS", str(3_000_000)))


class RestorePipeline:
    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        self.model_dir = ensure_dir(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._face_restorer: GFPGANer | None = None
        self._codeformer = CodeFormerRestorer(model_dir=self.model_dir, device=self.device)
        self._swinir = SwinIRPipeline(model_dir=self.model_dir, device=self.device)
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

    def restore_faces_gfpgan(self, image_bgr: np.ndarray, weight: float) -> np.ndarray:
        restorer = self._get_face_restorer()
        _, _, restored = restorer.enhance(
            image_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=weight,
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

    def _resolve_face_model(self, face_model: str, mode: str) -> str:
        if face_model != "auto":
            return face_model
        # Strong presets prefer CodeFormer when its weights are available.
        if mode in STRONG_MODES and CodeFormerRestorer.find_model_path(self.model_dir):
            return "codeformer"
        return "gfpgan"

    def _resolve_bg_model(self, bg_model: str, mode: str) -> str:
        if bg_model != "auto":
            return bg_model
        if mode in STRONG_MODES and SwinIRPipeline.find_model_path(self.model_dir):
            return "swinir"
        return "light"

    @staticmethod
    def _resolve_face_weight(face_model: str, mode: str, fidelity: float, face_weight: float | None) -> float:
        if face_weight is not None:
            return float(np.clip(face_weight, 0.0, 1.0))
        if face_model == "codeformer":
            # CodeFormer: low w = stronger restoration, high w = closer to input.
            if mode == "safe":
                return max(fidelity, 0.8)
            if mode in STRONG_MODES:
                return min(fidelity, 0.6)
            return fidelity
        # GFPGAN keeps the original preset behaviour.
        if mode == "safe":
            return min(fidelity, 0.5)
        if mode in STRONG_MODES:
            return max(fidelity, 0.85)
        return fidelity

    @staticmethod
    def _shrink_to_long_side(image_bgr: np.ndarray, long_side: int) -> np.ndarray:
        if long_side <= 0:
            return image_bgr
        h, w = image_bgr.shape[:2]
        current = max(h, w)
        if current <= long_side:
            return image_bgr
        scale = long_side / current
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        logger.info("shrinking input from %dx%d to %dx%d for processing", w, h, new_w, new_h)
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _cap_output_pixels(image_bgr: np.ndarray, max_pixels: int) -> np.ndarray:
        if max_pixels <= 0:
            return image_bgr
        h, w = image_bgr.shape[:2]
        if h * w <= max_pixels:
            return image_bgr
        scale = math.sqrt(max_pixels / (h * w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        logger.info("capping output from %dx%d to %dx%d (max %d pixels)", w, h, new_w, new_h, max_pixels)
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def process(
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
    ) -> Path:
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        if upscale not in (1, 2, 4):
            raise ValueError("upscale must be one of 1, 2, 4")
        if not 0.0 <= fidelity <= 1.0:
            raise ValueError("fidelity must be between 0.0 and 1.0")
        if face_model not in VALID_FACE_MODELS:
            raise ValueError(f"face_model must be one of {sorted(VALID_FACE_MODELS)}")
        if bg_model not in VALID_BG_MODELS:
            raise ValueError(f"bg_model must be one of {sorted(VALID_BG_MODELS)}")
        output_quality = int(np.clip(output_quality, 1, 100))

        image = load_image(input_path)
        result = self._shrink_to_long_side(image, MAX_INPUT_LONG_SIDE)

        resolved_face_model = self._resolve_face_model(face_model, mode)
        resolved_bg_model = self._resolve_bg_model(bg_model, mode)

        if mode in BACKGROUND_MODES:
            if resolved_bg_model == "swinir":
                result = self._swinir.restore(result, keep_size=True)
            elif resolved_bg_model == "light":
                if mode in STRONG_MODES:
                    result = self.enhance_background(result, 0.65)
                elif mode == "safe":
                    result = self.enhance_background(result, 0.2)
                else:
                    result = self.enhance_background(result, 0.35)

        if mode in FACE_MODES:
            applied_weight = self._resolve_face_weight(resolved_face_model, mode, fidelity, face_weight)
            if resolved_face_model == "codeformer":
                result = self._codeformer.restore(result, applied_weight)
            else:
                result = self.restore_faces_gfpgan(result, applied_weight)

        height, width = result.shape[:2]
        effective_upscale = resolve_effective_upscale(
            width,
            height,
            upscale,
            AUTO_2X_MAX_INPUT_PIXELS,
            dedicated_upscale_mode=mode == "upscale",
        )
        if effective_upscale != upscale:
            logger.info(
                "automatic upscale policy changed x%d to x%d for %dx%d input",
                upscale,
                effective_upscale,
                width,
                height,
            )

        if mode != "face" and effective_upscale > 1:
            result = self._upscale_pipeline.upscale(result, effective_upscale)

        result = self._cap_output_pixels(result, MAX_OUTPUT_PIXELS)

        output_path = Path(output_path)
        save_image(output_path, result, quality=output_quality)
        output_height, output_width = result.shape[:2]
        logger.info(
            "saved restored image %dx%d, %d bytes, quality=%d, upscale=x%d",
            output_width,
            output_height,
            output_path.stat().st_size,
            output_quality,
            effective_upscale,
        )
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
    face_model: str = "auto",
    face_weight: float | None = None,
    bg_model: str = "auto",
) -> Path:
    pipeline = get_pipeline(model_dir=model_dir)
    return pipeline.process(
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        upscale=upscale,
        fidelity=fidelity,
        output_quality=output_quality,
        face_model=face_model,
        face_weight=face_weight,
        bg_model=bg_model,
    )
