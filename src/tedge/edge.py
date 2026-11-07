"""ONNX export, int8 quantization and on-CPU benchmarking.

The benchmark mimics the device: batch size 1, one inference thread, latency measured
per window over many runs (p50 / p95), plus file size and accuracy of each artefact.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from .data import N


def export_onnx(model: torch.nn.Module, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 6, N)
    torch.onnx.export(
        model.eval(),
        (dummy,),
        str(path),
        input_names=["imu"],
        output_names=["logits"],
        dynamic_axes={"imu": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path


class _Calib:
    def __init__(self, x: np.ndarray) -> None:
        self._it = iter([{"imu": x[i : i + 1]} for i in range(len(x))])

    def get_next(self):
        return next(self._it, None)


def quantize_int8(src: Path, dst: Path, calib_x: np.ndarray) -> Path:
    """Static int8 quantization (QDQ) of conv/linear weights and activations.

    The calibration windows come from the training set only.
    """
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = dst.with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre))
    quantize_static(
        str(pre),
        str(dst),
        _Calib(calib_x.astype(np.float32)),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.Percentile,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        extra_options={"CalibPercentile": 99.99},
    )
    pre.unlink(missing_ok=True)
    return dst


def session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def onnx_predict(path: Path, x: np.ndarray, batch: int = 256) -> np.ndarray:
    s = session(path)
    out = [s.run(None, {"imu": x[i : i + batch].astype(np.float32)})[0] for i in range(0, len(x), batch)]
    return np.concatenate(out)


def latency_ms(fn, x: np.ndarray, runs: int = 500, warmup: int = 50) -> dict:
    """Batch-1 latency of fn(window) over ``runs`` windows."""
    for i in range(warmup):
        fn(x[i % len(x) : i % len(x) + 1])
    ts = []
    for i in range(runs):
        w = x[i % len(x) : i % len(x) + 1]
        t0 = time.perf_counter()
        fn(w)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95))}


def size_kb(path: Path) -> float:
    return os.path.getsize(path) / 1024
