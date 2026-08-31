from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.sizing import resolve_effective_upscale


class ResolveEffectiveUpscaleTests(unittest.TestCase):
    def test_skips_default_2x_for_six_megapixel_photo(self) -> None:
        self.assertEqual(
            resolve_effective_upscale(2968, 2084, 2, 3_000_000),
            1,
        )

    def test_keeps_default_2x_for_small_photo(self) -> None:
        self.assertEqual(
            resolve_effective_upscale(1500, 1500, 2, 3_000_000),
            2,
        )

    def test_keeps_explicit_upscale_requests(self) -> None:
        self.assertEqual(resolve_effective_upscale(4000, 3000, 1, 3_000_000), 1)
        self.assertEqual(resolve_effective_upscale(4000, 3000, 4, 3_000_000), 4)
        self.assertEqual(
            resolve_effective_upscale(
                4000,
                3000,
                2,
                3_000_000,
                dedicated_upscale_mode=True,
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
