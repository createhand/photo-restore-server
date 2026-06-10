from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from .utils import ensure_dir


class UpscalePipeline:
    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        self.model_dir = ensure_dir(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._upsamplers: dict[int, RealESRGANer] = {}

    def _create_model(self, outscale: int) -> RealESRGANer:
        if outscale not in (2, 4):
            raise ValueError("Real-ESRGAN supports only x2 or x4 in this server")

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=outscale,
        )

        model_path = self.model_dir / f"RealESRGAN_x{outscale}.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing Real-ESRGAN model file: {model_path}. Place the weight file in /app/models"
            )

        half = self.device.startswith("cuda")
        return RealESRGANer(
            scale=outscale,
            model_path=str(model_path),
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half,
            gpu_id=0 if half else None,
        )

    def get_upsampler(self, outscale: int) -> RealESRGANer:
        if outscale not in self._upsamplers:
            self._upsamplers[outscale] = self._create_model(outscale)
        return self._upsamplers[outscale]

    def upscale(self, image_bgr: np.ndarray, outscale: int) -> np.ndarray:
        if outscale == 1:
            return image_bgr
        upsampler = self.get_upsampler(outscale)
        output, _ = upsampler.enhance(image_bgr, outscale=outscale)
        return output


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def save_image(path: str | Path, image_bgr: np.ndarray) -> None:
    ensure_dir(Path(path).parent)
    success = cv2.imwrite(str(path), image_bgr)
    if not success:
        raise ValueError(f"Failed to save image: {path}")
