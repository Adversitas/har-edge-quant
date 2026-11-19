"""End-to-end study: DSP baseline vs CNN, then FP32 vs int8 vs pruned variants."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import compress, features, int8
from .data import ACTIVITIES, Split, Standardizer, load, subject_holdout, synthetic
from .model import TrainConfig, n_params, predict_logits, train


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def dsp_baseline(train_s: Split, test_s: Split) -> dict:
    ftr, fte = features.extract(train_s.x), features.extract(test_s.x)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    clf.fit(ftr, train_s.y)
    pred = clf.predict(fte)
    lr = clf[-1]
    size = (lr.coef_.size + lr.intercept_.size) * 4 + ftr.shape[1] * 2 * 4  # + scaler stats
    return {
        **scores(test_s.y, pred),
        "n_features": int(ftr.shape[1]),
        "model_bytes": int(size),
        "per_class_f1": f1_score(test_s.y, pred, average=None).round(4).tolist(),
    }


def run(
    data_dir: Path,
    reports: Path,
    seeds: list[int],
    epochs: int,
    use_synthetic: bool = False,
    calib_sizes: tuple[int, ...] = (16, 64, 256, 1024),
    log=print,
) -> dict:
    torch.set_num_threads(4)
    if use_synthetic:
        train_full, test_s = synthetic(360, seed=1), synthetic(180, seed=2)
    else:
        train_full, test_s = load(data_dir)
    log(f"train windows {len(train_full.y)}, test windows {len(test_s.y)} (subject-disjoint)")
    train_s, val_s = subject_holdout(train_full, n_val_subjects=4 if not use_synthetic else 2)
    std = Standardizer.fit(train_s.x)
    xtr, xva, xte = std(train_s.x), std(val_s.x), std(test_s.x)

    results: dict = {
        "dataset": "synthetic" if use_synthetic else "UCI HAR (raw inertial signals)",
        "n_train": int(len(train_s.y)),
        "n_val": int(len(val_s.y)),
        "n_test": int(len(test_s.y)),
        "epochs": epochs,
        "seeds": seeds,
        "platform": f"{platform.processor() or platform.machine()} / torch {torch.__version__}",
    }
    log("DSP features + logistic regression ...")
    results["dsp_baseline"] = dsp_baseline(train_full, test_s)

    per_seed = []
    calib_curve = {n: [] for n in calib_sizes}
    rng = np.random.default_rng(0)
    for seed in seeds:
        log(f"--- seed {seed}")
        model, hist = train(xtr, train_s.y, xva, val_s.y, TrainConfig(epochs=epochs, seed=seed))
        fp32_logits = predict_logits(model, xte)
        fp32_pred = fp32_logits.argmax(1)
        row: dict = {"seed": seed, "best_val_macro_f1": max(h["val_macro_f1"] for h in hist)}
        row["fp32"] = {
            **scores(test_s.y, fp32_pred),
            "params": n_params(model),
            "bytes": compress.serialized_bytes(model),
            "latency_ms": compress.latency_ms(model, xte[0]),
        }

        calib = xtr[rng.choice(len(xtr), size=min(1024, len(xtr)), replace=False)]
        tq = compress.ptq_int8(model, calib)
        tq_pred = predict_logits(tq, xte).argmax(1)
        row["torch_int8"] = {
            **scores(test_s.y, tq_pred),
            "bytes": compress.serialized_bytes(tq),
            "latency_ms": compress.latency_ms(tq, xte[0]),
            "agreement_with_fp32": float((tq_pred == fp32_pred).mean()),
        }

        folded = int8.fold_harnet(model)
        ref = int8.float_forward(folded, xte[:256])
        row["fold_max_abs_err"] = float(np.abs(ref - fp32_logits[:256]).max())
        for name, pct in (("int8_engine_p99.99", 99.99), ("int8_engine_max", 100.0)):
            qm = int8.quantize(folded, calib, percentile=pct)
            q_pred = qm.logits(xte).argmax(1)
            row[name] = {
                **scores(test_s.y, q_pred),
                "bytes": qm.weight_bytes(),
                "agreement_with_fp32": float((q_pred == fp32_pred).mean()),
                "layer_sqnr_db": [round(v, 2) for v in int8.layer_sqnr(folded, qm, xte[:512])],
            }
            if name == "int8_engine_p99.99":
                row["_confusion"] = confusion_matrix(test_s.y, q_pred).tolist()

        for n in calib_sizes:
            sub = xtr[rng.choice(len(xtr), size=min(n, len(xtr)), replace=False)]
            qm = int8.quantize(folded, sub, percentile=99.99)
            calib_curve[n].append(scores(test_s.y, qm.logits(xte).argmax(1))["macro_f1"])

        for amount in (0.5, 0.75, 0.9):
            pm = compress.magnitude_prune(model, amount)
            p_pred = predict_logits(pm, xte).argmax(1)
            row[f"pruned_{int(amount * 100)}"] = {
                **scores(test_s.y, p_pred),
                "sparsity": round(compress.sparsity(pm), 4),
            }
        per_seed.append(row)
        log(
            f"fp32 F1 {row['fp32']['macro_f1']:.4f}"
            f" | torch int8 {row['torch_int8']['macro_f1']:.4f}"
            f" | numpy int8 {row['int8_engine_p99.99']['macro_f1']:.4f}"
        )

    variants = [k for k in per_seed[0] if isinstance(per_seed[0][k], dict)]
    summary = {}
    for v in variants:
        agg = {}
        for metric in per_seed[0][v]:
            vals = [r[v][metric] for r in per_seed]
            if isinstance(vals[0], (int, float)):
                agg[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        summary[v] = agg
    results["summary"] = summary
    results["per_seed"] = per_seed
    results["calibration_size_macro_f1"] = {
        str(n): {"mean": float(np.mean(v)), "std": float(np.std(v))} for n, v in calib_curve.items()
    }
    results["activities"] = ACTIVITIES
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (reports / "results.md").write_text(render_markdown(results), encoding="utf-8")
    return results


def _fmt(s: dict, key: str, pct: bool = True) -> str:
    if key not in s:
        return "-"
    m, sd = s[key]["mean"], s[key]["std"]
    return f"{100 * m:.2f} ± {100 * sd:.2f}" if pct else f"{m:,.0f}"


def render_markdown(r: dict) -> str:
    s = r["summary"]
    b = r["dsp_baseline"]
    lines = [
        f"# Results ({r['dataset']})",
        "",
        f"Test set: {r['n_test']} windows from subjects never seen in training. "
        f"CNN numbers are mean ± std over seeds {r['seeds']} ({r['epochs']} epochs).",
        "",
        "| Model | Accuracy % | Macro-F1 % | Weights (bytes) | CPU latency, 1 window (ms) |",
        "|---|---|---|---|---|",
        f"| DSP features ({b['n_features']}) + logistic regression | {100 * b['accuracy']:.2f} | "
        f"{100 * b['macro_f1']:.2f} | {b['model_bytes']:,} | - |",
    ]
    names = {
        "fp32": "1D CNN, FP32 (PyTorch)",
        "torch_int8": "1D CNN, int8 PTQ (PyTorch x86 backend)",
        "int8_engine_p99.99": "1D CNN, int8 integer-only engine (NumPy, p99.99 calibration)",
        "int8_engine_max": "1D CNN, int8 integer-only engine (NumPy, max calibration)",
        "pruned_50": "1D CNN, 50% magnitude pruning (no fine-tuning)",
        "pruned_75": "1D CNN, 75% magnitude pruning (no fine-tuning)",
        "pruned_90": "1D CNN, 90% magnitude pruning (no fine-tuning)",
    }
    for k, label in names.items():
        if k not in s:
            continue
        lat = s[k].get("latency_ms")
        lat_s = f"{lat['mean']:.3f}" if lat else "-"
        lines.append(
            f"| {label} | {_fmt(s[k], 'accuracy')} | {_fmt(s[k], 'macro_f1')} | "
            f"{_fmt(s[k], 'bytes', pct=False)} | {lat_s} |"
        )
    lines += [
        "",
        "Agreement of int8 predictions with FP32 predictions: "
        f"PyTorch int8 {_fmt(s['torch_int8'], 'agreement_with_fp32')} %, "
        f"NumPy engine {_fmt(s['int8_engine_p99.99'], 'agreement_with_fp32')} %.",
        "",
        "## Calibration set size (NumPy int8 engine, macro-F1 %)",
        "",
        "| Calibration windows | Macro-F1 % |",
        "|---|---|",
    ]
    for n, v in r["calibration_size_macro_f1"].items():
        lines.append(f"| {n} | {100 * v['mean']:.2f} ± {100 * v['std']:.2f} |")
    sq = r["per_seed"][0]["int8_engine_p99.99"]["layer_sqnr_db"]
    lines += [
        "",
        "## Per-layer SQNR, seed 0 (dB, higher is better)",
        "",
        "| conv1 | conv2 | conv3 | global avg pool | fc (logits) |",
        "|---|---|---|---|---|",
        "| " + " | ".join(f"{v:.1f}" for v in sq) + " |",
        "",
        "## Confusion matrix, int8 engine, seed 0 (rows = truth)",
        "",
        "| | " + " | ".join(a[:4] for a in r["activities"]) + " |",
        "|---|" + "---|" * len(r["activities"]),
    ]
    for a, row in zip(r["activities"], r["per_seed"][0]["_confusion"], strict=True):
        lines.append(f"| {a} | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    lines.append(f"Platform: {r['platform']}")
    return "\n".join(lines) + "\n"
