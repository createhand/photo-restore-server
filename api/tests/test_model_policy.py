from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.model_policy import (
    choose_background_model,
    choose_face_model,
    choose_face_weight,
)


class ModelPolicyTests(unittest.TestCase):
    def test_auto_prefers_optional_restoration_models_when_available(self) -> None:
        self.assertEqual(choose_face_model("auto", True), "codeformer")
        self.assertEqual(choose_background_model("auto", True), "swinir")

    def test_auto_falls_back_when_optional_models_are_missing(self) -> None:
        self.assertEqual(choose_face_model("auto", False), "gfpgan")
        self.assertEqual(choose_background_model("auto", False), "light")

    def test_explicit_model_selection_is_preserved(self) -> None:
        self.assertEqual(choose_face_model("gfpgan", True), "gfpgan")
        self.assertEqual(choose_background_model("none", True), "none")

    def test_codeformer_defaults_to_identity_preserving_weight(self) -> None:
        self.assertEqual(
            choose_face_weight("codeformer", "enhance", 0.7, None),
            0.85,
        )
        self.assertEqual(
            choose_face_weight("codeformer", "full", 0.9, None),
            0.9,
        )

    def test_explicit_face_weight_takes_priority_and_is_clamped(self) -> None:
        self.assertEqual(
            choose_face_weight("codeformer", "full", 0.7, 0.95),
            0.95,
        )
        self.assertEqual(
            choose_face_weight("codeformer", "full", 0.7, 1.5),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
