from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from basicsr.archs.swinir_arch import SwinIR

from .utils import ensure_dir

MODEL_FILENAMES = (
    "SwinIR-M_x4_GAN.pth",
    "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-M_x4_GAN.pth",
)


class SwinIRPipeline:
    """Full-image restoration (denoise, deblur, JPEG artifact removal) with the
    real-world SwinIR-M x4 GAN model.

    The network natively upscales x4; `restore(keep_size=True)` resizes the
    result back to the input resolution so it acts as a pure restoration step
    before the final Real-ESRGAN upscale.
    """

    SCALE = 4
    WINDOW_SIZE = 8

    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        self.model_dir = ensure_dir(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tile = int(os.getenv("SWINIR_TILE", "256"))
        self.tile_overlap = int(os.getenv("SWINIR_TILE_OVERLAP", "32"))
        self._net: SwinIR | None = None

    @staticmethod
    def find_model_path(model_dir: str | Path) -> Path | None:
        for name in MODEL_FILENAMES:
            candidate = Path(model_dir) / name
            if candidate.exists():
                return candidate
        return None

    def _get_net(self) -> SwinIR:
        if self._net is None:
            model_path = self.find_model_path(self.model_dir)
            if model_path is None:
                raise FileNotFoundError(
                    f"Missing SwinIR model file: {self.model_dir / MODEL_FILENAMES[0]}. "
                    "Download 003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-M_x4_GAN.pth from "
                    "https://github.com/JingyunLiang/SwinIR/releases and place it in /app/models"
                )

            net = SwinIR(
                upscale=self.SCALE,
                in_chans=3,
                img_size=64,
                window_size=self.WINDOW_SIZE,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="1conv",
            )
            checkpoint = torch.load(model_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("params_ema") or checkpoint.get("params") or checkpoint
            else:
                state_dict = checkpoint
            net.load_state_dict(state_dict, strict=True)
            net.eval()
            self._net = net.to(self.device)
        return self._net

    def _inference(self, img: torch.Tensor) -> torch.Tensor:
        net = self._get_net()
        _, _, h, w = img.size()
        if self.tile <= 0 or (h <= self.tile and w <= self.tile):
            return net(img)

        tile = min(self.tile, h, w)
        tile = max(self.WINDOW_SIZE, tile - tile % self.WINDOW_SIZE)
        stride = max(1, tile - self.tile_overlap)
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

        scale = self.SCALE
        output = torch.zeros(
            img.shape[0], img.shape[1], h * scale, w * scale,
            dtype=img.dtype, device=img.device,
        )
        weights = torch.zeros_like(output)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = net(in_patch)
                output[
                    ...,
                    h_idx * scale:(h_idx + tile) * scale,
                    w_idx * scale:(w_idx + tile) * scale,
                ].add_(out_patch)
                weights[
                    ...,
                    h_idx * scale:(h_idx + tile) * scale,
                    w_idx * scale:(w_idx + tile) * scale,
                ].add_(torch.ones_like(out_patch))
        return output.div_(weights)

    def restore(self, image_bgr: np.ndarray, keep_size: bool = True) -> np.ndarray:
        img = image_bgr.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1)))
        img = img.unsqueeze(0).to(self.device)

        _, _, h_old, w_old = img.size()
        window = self.WINDOW_SIZE
        h_pad = (h_old // window + 1) * window - h_old
        w_pad = (w_old // window + 1) * window - w_old
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h_old + h_pad, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, : w_old + w_pad]

        with torch.no_grad():
            out = self._inference(img)
        out = out[..., : h_old * self.SCALE, : w_old * self.SCALE]
        out = out.squeeze(0).float().clamp_(0, 1).cpu().numpy()
        out = np.transpose(out[[2, 1, 0], :, :], (1, 2, 0))
        result = (out * 255.0).round().astype(np.uint8)

        if keep_size:
            result = cv2.resize(result, (w_old, h_old), interpolation=cv2.INTER_AREA)
        return result
