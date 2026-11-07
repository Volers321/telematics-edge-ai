"""1D CNNs for 4-second IMU windows, sized for an edge CPU.

Two input variants:

* ``raw``: the six device-frame channels, scaled. The network has to learn that the box
  can be mounted any way round; random-rotation augmentation helps it.
* ``invariant``: the window is first turned into four mounting-invariant channels
  (vertical acceleration, horizontal magnitude, yaw rate, vibration) by ``Invariant``,
  the PyTorch twin of ``features.decompose``. Because the transform is a module, it is
  exported into the same ONNX graph and the device feeds raw sensor data.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .data import FS


class Invariant(nn.Module):
    """[B, 6, N] raw IMU -> [B, 4, N]: vert, h_mag, yaw, vib (see features.decompose)."""

    def __init__(self) -> None:
        super().__init__()
        wf, wv = int(0.3 * FS), 5
        self.register_buffer("k_force3", torch.full((3, 1, wf), 1.0 / wf))
        self.register_buffer("k_force1", torch.full((1, 1, wf), 1.0 / wf))
        self.register_buffer("k_vib3", torch.full((3, 1, wv), 1.0 / wv))

    @staticmethod
    def _avg(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv1d(x, k, padding=k.shape[-1] // 2, groups=k.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, w = x[:, :3], x[:, 3:]
        mean = a.mean(dim=2, keepdim=True)
        g_hat = mean / torch.sqrt((mean * mean).sum(dim=1, keepdim=True))
        lin = a - mean
        vert = (lin * g_hat).sum(dim=1, keepdim=True)
        horiz = self._avg(lin - vert * g_hat, self.k_force3)
        h_mag = torch.sqrt((horiz * horiz).sum(dim=1, keepdim=True))
        yaw = self._avg((w * g_hat).sum(dim=1, keepdim=True), self.k_force1)
        hf = lin - self._avg(lin, self.k_vib3)
        vib = torch.sqrt((hf * hf).sum(dim=1, keepdim=True))
        return torch.cat([vert, h_mag, yaw, vib], dim=1)


class IMUNet(nn.Module):
    def __init__(self, in_channels: int = 6, n_classes: int = 5, width: int = 32) -> None:
        super().__init__()

        def block(i: int, o: int, k: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())

        self.features = nn.Sequential(
            block(in_channels, width, 7),
            nn.MaxPool1d(2),
            block(width, 2 * width, 5),
            nn.MaxPool1d(2),
            block(2 * width, 2 * width, 5),
            nn.MaxPool1d(2),
            block(2 * width, 2 * width, 3),
        )
        self.head = nn.Linear(4 * width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return self.head(torch.cat([f.mean(dim=2), f.amax(dim=2)], dim=1))


class EdgeModel(nn.Module):
    """Raw device-frame IMU in, class logits out. This is what gets exported."""

    def __init__(self, net: IMUNet, scale: torch.Tensor, invariant: bool) -> None:
        super().__init__()
        self.invariant = invariant
        self.pre = Invariant() if invariant else nn.Identity()
        self.register_buffer("scale", scale.reshape(1, -1, 1).float())
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.pre(x) / self.scale)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def random_rotation(rng: np.random.Generator, batch: int) -> np.ndarray:
    """Uniformly random proper 3D rotations, shape [B, 3, 3]."""
    q, r = np.linalg.qr(rng.normal(size=(batch, 3, 3)))
    q = q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    q[:, :, 0] *= np.linalg.det(q)[:, None]
    return q


def rotate_batch(x: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Rotate accelerometer and gyroscope together: the same motion, mounted differently."""
    a = np.einsum("bij,bjn->bin", rot, x[:, :3])
    w = np.einsum("bij,bjn->bin", rot, x[:, 3:])
    return np.concatenate([a, w], axis=1).astype(np.float32)


@torch.no_grad()
def to_invariant(x: np.ndarray, batch: int = 1024) -> np.ndarray:
    pre = Invariant()
    return np.concatenate([pre(torch.from_numpy(x[i : i + batch])).numpy() for i in range(0, len(x), batch)])


def _scale(x: np.ndarray, invariant: bool) -> torch.Tensor:
    if invariant:
        return torch.from_numpy(x.std(axis=(0, 2)).astype(np.float32))
    # accelerometer and gyroscope scaled as groups, so a rotation stays a rotation
    a, w = float(x[:, :3].std()), float(x[:, 3:].std())
    return torch.tensor([a, a, a, w, w, w])


def train(
    x: np.ndarray,
    y: np.ndarray,
    *,
    invariant: bool = True,
    augment: bool = False,
    epochs: int = 20,
    batch: int = 128,
    lr: float = 3e-3,
    seed: int = 0,
    log=print,
) -> EdgeModel:
    """Train on raw windows ``x`` [B, 6, N]; returns an EdgeModel that takes raw windows."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    inp = to_invariant(x) if invariant else x
    scale = _scale(inp, invariant)
    net = IMUNet(in_channels=inp.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(x) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    lossf = nn.CrossEntropyLoss()
    s = scale.reshape(1, -1, 1)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(x))
        total = 0.0
        for i in range(0, len(x), batch):
            j = order[i : i + batch]
            if len(j) < 2:  # BatchNorm needs more than one sample
                continue
            xb = inp[j]
            if augment and not invariant:
                xb = rotate_batch(xb, random_rotation(rng, len(j)))
            loss = lossf(net(torch.from_numpy(xb) / s), torch.from_numpy(y[j]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(j)
        if ep % 5 == 4 or ep == epochs - 1:
            log(f"  epoch {ep + 1:3d}  loss {total / len(x):.4f}")
    net.eval()
    return EdgeModel(net, scale, invariant).eval()


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    return np.concatenate(
        [model(torch.from_numpy(x[i : i + batch])).numpy() for i in range(0, len(x), batch)]
    )


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    return predict_logits(model, x).argmax(1)
