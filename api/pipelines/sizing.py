from __future__ import annotations


def resolve_effective_upscale(
    width: int,
    height: int,
    requested_upscale: int,
    max_2x_input_pixels: int,
    *,
    dedicated_upscale_mode: bool = False,
) -> int:
    """Skip the default 2x pass when the source already has enough pixels.

    Explicit 1x/4x requests and the dedicated upscale mode keep their original
    meaning. The automatic policy only adjusts the normal restoration preset's
    default 2x request.
    """
    if requested_upscale != 2 or dedicated_upscale_mode or max_2x_input_pixels <= 0:
        return requested_upscale

    input_pixels = max(0, width) * max(0, height)
    return 2 if input_pixels <= max_2x_input_pixels else 1
