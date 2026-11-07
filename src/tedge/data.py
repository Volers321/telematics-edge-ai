"""Synthetic IMU windows from a telematics box mounted at an unknown angle.

Each sample is 4 s of a 6-axis IMU at 50 Hz (accelerometer x/y/z in m/s^2, gyroscope
x/y/z in rad/s) in the *device* frame. The physics is simulated in the vehicle frame
(x forward, y left, z up) and then rotated by the device's mounting, which is fixed per
device and unknown to the model:

* yaw of the mounting: anything from 0 to 360 degrees (installers put the box any way
  round), pitch and roll up to +-15 degrees;
* per-device accelerometer bias and noise level, engine vibration frequency.

Labels follow fleet-telematics practice: an event is *harsh* only above a threshold
(e.g. braking stronger than 0.35 g). Events below the threshold are generated too and
labelled ``normal``, so a model has to learn the threshold, not just "something
happened". Braking and acceleration look the same in a single window if you do not know
which way is forward: the body pitch follows the inertial force in both cases, so it
does not help. What tells them apart is what the box does measure: the engine vibrates
harder under throttle and goes quiet when the driver brakes, and harsh braking is
usually shorter than harsh acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 50  # Hz
WINDOW_S = 4.0
N = int(FS * WINDOW_S)
G = 9.81

CLASSES = ["normal", "harsh_braking", "harsh_acceleration", "harsh_cornering", "road_bump"]
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]

# Thresholds in m/s^2 (peak), typical of fleet scoring rules.
THRESHOLDS = {"braking": 0.35 * G, "acceleration": 0.30 * G, "cornering": 0.40 * G, "bump": 6.0}

# Class mix of generated windows: what kind of manoeuvre happens in the window.
MANOEUVRES = {"cruise": 0.20, "braking": 0.20, "acceleration": 0.20, "cornering": 0.20, "bump": 0.20}


@dataclass(frozen=True)
class Device:
    id: int
    rotation: np.ndarray  # 3x3, vehicle -> device
    bias: np.ndarray  # accelerometer bias, m/s^2
    noise: float  # accelerometer noise std, m/s^2
    gyro_noise: float
    engine_hz: float


def _rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy, cp, sp, cr, sr = (
        np.cos(yaw),
        np.sin(yaw),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(roll),
        np.sin(roll),
    )
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def make_devices(n: int, seed: int) -> list[Device]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        r = _rot(
            rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(-15, 15)), np.radians(rng.uniform(-15, 15))
        )
        out.append(
            Device(
                id=i,
                rotation=r,
                bias=rng.normal(0, 0.15, 3),
                noise=rng.uniform(0.05, 0.20),
                gyro_noise=rng.uniform(0.002, 0.01),
                engine_hz=rng.uniform(12, 22),
            )
        )
    return out


def _pulse(t: np.ndarray, start: float, dur: float) -> np.ndarray:
    """Half-sine pulse with peak 1 between start and start + dur."""
    x = (t - start) / dur
    return np.where((x >= 0) & (x <= 1), np.sin(np.pi * np.clip(x, 0, 1)), 0.0)


def _smooth_noise(rng: np.random.Generator, std: float, width: int = 25) -> np.ndarray:
    k = np.hanning(width)
    k /= k.sum()
    x = np.convolve(rng.normal(0, 1, N + width), k, mode="same")[:N]
    return x / (x.std() + 1e-9) * std


def simulate(device: Device, rng: np.random.Generator) -> tuple[np.ndarray, int, dict]:
    """One window -> (array [6, N] in device frame, class index, metadata)."""
    t = np.arange(N) / FS
    kind = rng.choice(list(MANOEUVRES), p=list(MANOEUVRES.values()))
    acc = np.zeros((3, N))  # vehicle frame, without gravity
    gyr = np.zeros((3, N))
    # background driving: gentle speed changes and steering
    acc[0] += _smooth_noise(rng, rng.uniform(0.1, 0.5))
    acc[1] += _smooth_noise(rng, rng.uniform(0.1, 0.5))
    acc[2] += _smooth_noise(rng, 0.15, width=5)  # road texture
    gyr[2] += _smooth_noise(rng, 0.02)
    throttle = np.full(N, rng.uniform(0.2, 0.5))

    label, peak = 0, 0.0
    dur = {"braking": rng.uniform(0.6, 1.6), "acceleration": rng.uniform(1.2, 2.4)}.get(
        kind, rng.uniform(0.8, 1.8)
    )
    start = rng.uniform(0.2, WINDOW_S - dur - 0.2)
    p = _pulse(t, start, dur)
    if kind == "braking":
        peak = rng.uniform(1.0, 7.5)
        acc[0] -= peak * p
        gyr[1] += 0.012 * peak * np.gradient(p, 1 / FS) * dur  # body pitch follows the force
        throttle = throttle * (1 - p)
        label = 1 if peak >= THRESHOLDS["braking"] else 0
    elif kind == "acceleration":
        peak = rng.uniform(0.8, 5.5)
        acc[0] += peak * p
        gyr[1] -= 0.012 * peak * np.gradient(p, 1 / FS) * dur
        throttle = np.clip(throttle + p, 0, 1.2)
        label = 2 if peak >= THRESHOLDS["acceleration"] else 0
    elif kind == "cornering":
        peak = rng.uniform(1.0, 8.0) * rng.choice([-1, 1])
        speed = rng.uniform(8, 25)  # m/s
        acc[1] += peak * p
        gyr[2] += peak / speed * p  # a_lat = v * yaw_rate
        gyr[0] -= 0.01 * peak * p  # body roll
        label = 3 if abs(peak) >= THRESHOLDS["cornering"] else 0
        peak = abs(peak)
    elif kind == "bump":
        peak = rng.uniform(1.5, 16.0)
        hit = rng.uniform(0.3, 3.5)
        tau = t - hit
        f = rng.uniform(7, 12)
        ring = np.where(tau >= 0, np.exp(-tau * rng.uniform(6, 12)) * np.sin(2 * np.pi * f * tau), 0.0)
        acc[2] += peak * ring
        acc[0] -= 0.15 * peak * np.abs(ring)
        gyr[1] += 0.02 * peak * ring
        label = 4 if peak >= THRESHOLDS["bump"] else 0

    vib = 0.08 + 0.35 * throttle
    acc[2] += vib * np.sin(2 * np.pi * device.engine_hz * t + rng.uniform(0, 2 * np.pi))
    acc[0] += 0.3 * vib * np.sin(2 * np.pi * device.engine_hz * t)

    grav = np.array([0, 0, G])[:, None]
    meas_a = device.rotation @ (acc + grav) + device.bias[:, None] + rng.normal(0, device.noise, (3, N))
    meas_g = device.rotation @ gyr + rng.normal(0, device.gyro_noise, (3, N))
    x = np.vstack([meas_a, meas_g]).astype(np.float32)
    return x, label, {"kind": kind, "peak": float(peak), "device": device.id}


def generate(n_devices: int, windows_per_device: int, seed: int) -> dict[str, np.ndarray]:
    devices = make_devices(n_devices, seed)
    rng = np.random.default_rng(seed + 1)
    xs, ys, dev, kinds, peaks = [], [], [], [], []
    for d in devices:
        for _ in range(windows_per_device):
            x, y, m = simulate(d, rng)
            xs.append(x)
            ys.append(y)
            dev.append(d.id)
            kinds.append(m["kind"])
            peaks.append(m["peak"])
    return {
        "x": np.stack(xs),
        "y": np.array(ys, dtype=np.int64),
        "device": np.array(dev),
        "kind": np.array(kinds),
        "peak": np.array(peaks, dtype=np.float32),
    }


def split_by_device(ds: dict, test_frac: float = 0.25, seed: int = 0) -> tuple[dict, dict]:
    """Test devices are never seen in training: new installs, new mounting angles."""
    ids = np.unique(ds["device"])
    rng = np.random.default_rng(seed)
    test_ids = rng.choice(ids, size=max(1, int(len(ids) * test_frac)), replace=False)
    m = np.isin(ds["device"], test_ids)
    pick = lambda mask: {k: v[mask] for k, v in ds.items()}  # noqa: E731
    return pick(~m), pick(m)


def split_random(ds: dict, test_frac: float = 0.25, seed: int = 0) -> tuple[dict, dict]:
    """The leaky split: windows from the same box on both sides."""
    rng = np.random.default_rng(seed)
    m = rng.random(len(ds["y"])) < test_frac
    pick = lambda mask: {k: v[mask] for k, v in ds.items()}  # noqa: E731
    return pick(~m), pick(m)
