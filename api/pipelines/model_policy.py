from __future__ import annotations


STRONG_MODES = {"pretty", "enhance"}


def clamp_weight(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def choose_face_model(requested_model: str, codeformer_available: bool) -> str:
    if requested_model != "auto":
        return requested_model
    return "codeformer" if codeformer_available else "gfpgan"


def choose_background_model(requested_model: str, swinir_available: bool) -> str:
    if requested_model != "auto":
        return requested_model
    return "swinir" if swinir_available else "light"


def choose_face_weight(
    face_model: str,
    mode: str,
    fidelity: float,
    requested_weight: float | None,
    default_codeformer_weight: float = 0.85,
) -> float:
    if requested_weight is not None:
        return clamp_weight(requested_weight)

    fidelity = clamp_weight(fidelity)
    if face_model == "codeformer":
        # CodeFormer uses a higher value to retain more of the original identity.
        return max(fidelity, clamp_weight(default_codeformer_weight))

    # Preserve the existing GFPGAN presets for installations without CodeFormer.
    if mode == "safe":
        return min(fidelity, 0.5)
    if mode in STRONG_MODES:
        return max(fidelity, 0.85)
    return fidelity
