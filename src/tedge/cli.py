"""Command line: ``tedge <command>``.

tedge run              # full experiment: data, 5 models, leakage check, edge export, charts
tedge run --quick      # small version for CI (a minute on a laptop)
tedge predict FILE.npy # classify windows [B, 6, 200] with the exported int8 model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import data

RESULTS = Path("results")
DOCS = Path("docs")
ARTIFACTS = Path("artifacts")


def _scores(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {"accuracy": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro"))}


def _mean_std(runs: list[dict]) -> dict:
    out = {}
    for k in runs[0]:
        v = np.array([r[k] for r in runs])
        out[k] = float(v.mean())
        out[k + "_std"] = float(v.std())
    return out


def _near_threshold(te: dict, p: np.ndarray) -> dict:
    """Accuracy by how close the event's strength was to its harshness threshold."""
    kind_thr = dict(data.THRESHOLDS)
    pairs = zip(te["kind"], te["peak"], strict=True)
    rel = np.array([abs(pk / kind_thr[k] - 1) if k in kind_thr else np.nan for k, pk in pairs])
    ok = p == te["y"]
    bands = {
        "within 10% of threshold": rel < 0.10,
        "10-25%": (rel >= 0.10) & (rel < 0.25),
        "more than 25%": rel >= 0.25,
        "cruise (no event)": np.isnan(rel),
    }
    return {
        k: {"windows": int(m.sum()), "accuracy": float(ok[m].mean()) if m.any() else None}
        for k, m in bands.items()
    }


def cmd_run(args) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import confusion_matrix

    from . import edge, features, model

    n_dev, per_dev = (24, 80) if args.quick else (args.devices, args.windows)
    epochs = 4 if args.quick else args.epochs
    seeds = [0] if args.quick else list(range(args.seeds))
    t0 = time.time()
    ds = data.generate(n_dev, per_dev, seed=args.data_seed)
    tr, te = data.split_by_device(ds, seed=args.data_seed)
    print(
        f"{len(ds['y'])} windows from {n_dev} boxes; train {len(tr['y'])} "
        f"({len(np.unique(tr['device']))} boxes), test {len(te['y'])} "
        f"({len(np.unique(te['device']))} unseen boxes)"
    )
    out: dict = {
        "data": {
            "devices": n_dev,
            "windows": int(len(ds["y"])),
            "train": int(len(tr["y"])),
            "test": int(len(te["y"])),
            "class_counts_test": np.bincount(te["y"], minlength=5).tolist(),
            "classes": data.CLASSES,
        },
        "models": {},
    }

    out["models"]["rule"] = _scores(te["y"], features.rule_baseline(te["x"]))
    print(f"rule baseline        acc {out['models']['rule']['accuracy']:.3f}")

    ftr, names = features.features(tr["x"])
    fte, _ = features.features(te["x"])
    runs, gbm = [], None
    for s in seeds:
        gbm = HistGradientBoostingClassifier(max_iter=300, random_state=s).fit(ftr, tr["y"])
        runs.append(_scores(te["y"], gbm.predict(fte)))
    out["models"]["gbm_features"] = _mean_std(runs)
    print(f"GBM on {len(names)} features  acc {out['models']['gbm_features']['accuracy']:.3f}")

    variants = {
        "cnn_raw": dict(invariant=False, augment=False),
        "cnn_raw_rot_aug": dict(invariant=False, augment=True),
        "cnn_invariant": dict(invariant=True, augment=False),
    }
    best = None
    for name, kw in variants.items():
        runs = []
        for s in seeds:
            m = model.train(tr["x"], tr["y"], epochs=epochs, seed=s, log=lambda *_: None, **kw)
            p = model.predict(m, te["x"])
            runs.append(_scores(te["y"], p))
            if name == "cnn_invariant" and s == seeds[0]:
                best, best_p = m, p
        out["models"][name] = _mean_std(runs)
        r = out["models"][name]
        print(f"{name:20s} acc {r['accuracy']:.3f} ± {r['accuracy_std']:.3f}")

    out["best_model"] = {
        "name": "cnn_invariant",
        "params": model.n_params(best),
        "confusion": confusion_matrix(te["y"], best_p, labels=range(5)).tolist(),
        "by_distance_to_threshold": _near_threshold(te, best_p),
    }

    # the leaky split: same boxes on both sides
    ltr, lte = data.split_random(ds, seed=args.data_seed)
    m = model.train(ltr["x"], ltr["y"], epochs=epochs, invariant=False, log=lambda *_: None)
    out["leakage"] = {
        "cnn_raw_random_split": _scores(lte["y"], model.predict(m, lte["x"])),
        "cnn_raw_device_split": {k: out["models"]["cnn_raw"][k] for k in ("accuracy", "macro_f1")},
    }
    print(f"leaky random split   acc {out['leakage']['cnn_raw_random_split']['accuracy']:.3f}")

    # edge export of the invariant CNN
    ARTIFACTS.mkdir(exist_ok=True)
    fp32 = edge.export_onnx(best, ARTIFACTS / "tedge_fp32.onnx")
    int8 = edge.quantize_int8(fp32, ARTIFACTS / "tedge_int8.onnx", tr["x"][: 300 if not args.quick else 64])
    import torch

    torch.set_num_threads(1)
    runs = 200 if args.quick else 1000
    rows = {}
    with torch.no_grad():
        rows["pytorch_fp32"] = {
            **_scores(te["y"], best_p),
            **edge.latency_ms(lambda w: best(torch.from_numpy(w)), te["x"], runs),
        }
    for tag, path in (("onnx_fp32", fp32), ("onnx_int8", int8)):
        sess = edge.session(path)
        p = edge.onnx_predict(path, te["x"]).argmax(1)
        rows[tag] = {
            **_scores(te["y"], p),
            "size_kb": edge.size_kb(path),
            "agreement_with_pytorch": float((p == best_p).mean()),
            **edge.latency_ms(lambda w, s=sess: s.run(None, {"imu": w}), te["x"], runs),
        }
    rows["gbm_features"] = {
        **{k: out["models"]["gbm_features"][k] for k in ("accuracy", "macro_f1")},
        **edge.latency_ms(lambda w: gbm.predict(features.features(w)[0]), te["x"], min(runs, 300)),
    }
    out["edge"] = rows
    for k, r in rows.items():
        print(
            f"{k:14s} acc {r['accuracy']:.3f}  p50 {r['p50_ms']:.3f} ms  p95 {r['p95_ms']:.3f} ms"
            + (f"  {r['size_kb']:.0f} KB" if "size_kb" in r else "")
        )
    out["runtime_s"] = round(time.time() - t0, 1)

    if not args.quick:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "results.json").write_text(json.dumps(out, indent=1), "utf-8")
        from . import report

        report.charts(out, DOCS)
        print(f"-> {RESULTS / 'results.json'}, charts in {DOCS}/")
    return out


def cmd_predict(args) -> None:
    from . import edge

    x = np.load(args.file).astype(np.float32)
    if x.ndim == 2:
        x = x[None]
    logits = edge.onnx_predict(Path(args.model), x)
    for i, row in enumerate(logits):
        e = np.exp(row - row.max())
        pr = e / e.sum()
        k = int(pr.argmax())
        print(f"window {i}: {data.CLASSES[k]} ({pr[k]:.0%})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tedge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="full experiment")
    r.add_argument("--quick", action="store_true")
    r.add_argument("--devices", type=int, default=80)
    r.add_argument("--windows", type=int, default=150, help="windows per device")
    r.add_argument("--epochs", type=int, default=20)
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--data-seed", type=int, default=1)
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("predict", help="classify windows from a .npy file")
    p.add_argument("file")
    p.add_argument("--model", default=str(ARTIFACTS / "tedge_int8.onnx"))
    p.set_defaults(fn=cmd_predict)
    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
