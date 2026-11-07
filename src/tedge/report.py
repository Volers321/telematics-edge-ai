"""Charts for the README."""

from __future__ import annotations

from pathlib import Path

LABELS = {
    "rule": "threshold rule",
    "gbm_features": "GBM, 13 physics features",
    "cnn_raw": "CNN, raw axes",
    "cnn_raw_rot_aug": "CNN, raw + rotation aug.",
    "cnn_invariant": "CNN, invariant channels",
}


def charts(out: dict, docs: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs.mkdir(exist_ok=True)
    m = out["models"]
    keys = list(LABELS)
    vals = [m[k]["macro_f1"] for k in keys]
    errs = [m[k].get("macro_f1_std", 0) for k in keys]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh([LABELS[k] for k in keys], vals, xerr=errs, color="#4C6EF5", capsize=3)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.5, 1.02)
    ax.invert_yaxis()
    ax.set_xlabel("macro-F1 on boxes never seen in training (mean ± std over 3 seeds)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(docs / "models.png", dpi=150)
    plt.close(fig)

    e = out["edge"]
    keys = ["pytorch_fp32", "onnx_fp32", "onnx_int8", "gbm_features"]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    p50 = [e[k]["p50_ms"] for k in keys]
    p95 = [e[k]["p95_ms"] for k in keys]
    y = range(len(keys))
    ax.barh(y, p95, color="#CED4DA", label="p95")
    ax.barh(y, p50, color="#2F9E44", label="p50")
    ax.set_yticks(list(y), ["PyTorch fp32", "ONNX Runtime fp32", "ONNX Runtime int8", "GBM + features"])
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("latency per 4 s window, batch 1, 1 CPU thread (ms, log scale)")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(docs / "latency.png", dpi=150)
    plt.close(fig)
