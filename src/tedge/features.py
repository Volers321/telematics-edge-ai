"""Mounting-invariant signal processing shared by the rule baseline and the GBM.

The box does not know how it is mounted, so everything is expressed relative to the
gravity direction estimated from the window itself:

* vertical = linear acceleration along gravity (bumps),
* horizontal = what is left (braking, acceleration, cornering), as a magnitude,
* yaw rate = gyroscope along gravity (turning).
"""

from __future__ import annotations

import numpy as np

from .data import FS


def _movavg(x: np.ndarray, width: int) -> np.ndarray:
    k = np.ones(width) / width
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), -1, x)


def decompose(x: np.ndarray) -> dict[str, np.ndarray]:
    """x: [B, 6, N] -> components, each [B, N]."""
    a, w = x[:, :3].astype(np.float64), x[:, 3:].astype(np.float64)
    mean = a.mean(axis=2, keepdims=True)
    g_hat = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    lin = a - mean
    vert = (lin * g_hat).sum(axis=1)
    horiz = lin - vert[:, None] * g_hat
    h_s = _movavg(horiz, int(0.3 * FS))  # remove engine vibration
    yaw = (w * g_hat).sum(axis=1)
    hf = lin - _movavg(lin, 5)  # vibration band
    return {
        "vert": vert,
        "h_mag": np.linalg.norm(h_s, axis=1),
        "yaw": _movavg(yaw[:, None], int(0.3 * FS))[:, 0],
        "vib": np.linalg.norm(hf, axis=1),
    }


def _q(x: np.ndarray, q: float) -> np.ndarray:
    return np.quantile(x, q, axis=1)


def features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    c = decompose(x)
    h, v, yaw, vib = c["h_mag"], c["vert"], c["yaw"], c["vib"]
    peak_i = h.argmax(axis=1)
    idx = np.arange(len(h))
    # engine vibration while the horizontal force is high vs while it is low
    hi = h > 0.5 * h.max(axis=1, keepdims=True)
    vib_hi = (vib * hi).sum(1) / np.maximum(hi.sum(1), 1)
    vib_lo = (vib * ~hi).sum(1) / np.maximum((~hi).sum(1), 1)
    cols = {
        "h_peak": h.max(1),
        "h_p90": _q(h, 0.9),
        "h_mean": h.mean(1),
        "h_dur_2ms2": (h > 2.0).sum(1) / FS,
        "h_dur_half": hi.sum(1) / FS,
        "yaw_peak": np.abs(yaw).max(1),
        "yaw_at_h_peak": np.abs(yaw[idx, peak_i]),
        "v_ptp": v.max(1) - v.min(1),
        "v_peak": np.abs(v).max(1),
        "v_std": v.std(1),
        "v_jerk": np.abs(np.diff(v, axis=1)).max(1) * FS,
        "vib_mean": vib.mean(1),
        "vib_ratio": vib_hi / np.maximum(vib_lo, 1e-6),
    }
    names = list(cols)
    return np.column_stack([cols[k] for k in names]).astype(np.float32), names


def rule_baseline(x: np.ndarray) -> np.ndarray:
    """What a simple firmware rule does: thresholds on mounting-invariant peaks.

    It cannot tell braking from acceleration (no forward axis), so any strong
    longitudinal event is called braking.
    """
    from .data import THRESHOLDS

    c = decompose(x)
    h_peak = c["h_mag"].max(1)
    yaw_peak = np.abs(c["yaw"]).max(1)
    v_peak = np.abs(c["vert"]).max(1)
    out = np.zeros(len(x), dtype=np.int64)
    out[h_peak >= THRESHOLDS["braking"]] = 1
    out[(h_peak >= THRESHOLDS["cornering"]) & (yaw_peak > 0.15)] = 3
    out[v_peak >= THRESHOLDS["bump"]] = 4
    return out
