from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from .archs.codeformer_arch import CodeFormer
from .utils import ensure_dir

logger = logging.getLogger("photo-restore-server")

MODEL_FILENAMES = ("CodeFormer.pth", "codeformer.pth")


class CodeFormerRestorer:
    """Blind face restoration with CodeFormer.

    Detection/alignment/paste-back uses the same facexlib helper GFPGAN uses,
    so both face models behave identically outside the per-face network call.
    """

    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        self.model_dir = ensure_dir(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._net: CodeFormer | None = None
        self._face_helper: FaceRestoreHelper | None = None

    @staticmethod
    def find_model_path(model_dir: str | Path) -> Path | None:
        for name in MODEL_FILENAMES:
            candidate = Path(model_dir) / name
            if candidate.exists():
                return candidate
        return None

    def _get_net(self) -> CodeFormer:
        if self._net is None:
            model_path = self.find_model_path(self.model_dir)
            if model_path is None:
                raise FileNotFoundError(
                    f"Missing CodeFormer model file: {self.model_dir / MODEL_FILENAMES[0]}. "
                    "Download codeformer.pth from "
                    "https://github.com/sczhou/CodeFormer/releases and place it in /app/models"
                )

            net = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            )
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint.get("params_ema", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            net.load_state_dict(state_dict)
            net.eval()
            self._net = net.to(self.device)
        return self._net

    def _get_face_helper(self) -> FaceRestoreHelper:
        if self._face_helper is None:
            self._face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=True,
                device=self.device,
                model_rootpath=str(self.model_dir),
            )
        return self._face_helper

    def restore(self, image_bgr: np.ndarray, weight: float) -> np.ndarray:
        """Restore all faces in the image. weight: 0.0 = strongest restoration, 1.0 = closest to input."""
        weight = float(np.clip(weight, 0.0, 1.0))
        net = self._get_net()
        helper = self._get_face_helper()
        helper.clean_all()
        helper.read_image(image_bgr)
        num_faces = helper.get_face_landmarks_5(
            only_center_face=False,
            resize=640,
            eye_dist_threshold=5,
        )
        if num_faces == 0:
            logger.info("CodeFormer: no face detected, returning input unchanged")
            return image_bgr
        helper.align_warp_face()

        for cropped_face in helper.cropped_faces:
            face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_t = face_t.unsqueeze(0).to(self.device)
            try:
                with torch.no_grad():
                    output = net(face_t, w=weight, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
            except RuntimeError:
                logger.exception("CodeFormer inference failed for a face, keeping original crop")
                restored_face = cropped_face
            helper.add_restored_face(np.asarray(restored_face, dtype=np.uint8))

        helper.get_inverse_affine(None)
        return helper.paste_faces_to_input_image(upsample_img=None)
