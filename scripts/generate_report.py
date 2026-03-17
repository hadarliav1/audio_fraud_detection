#!/usr/bin/env python3
"""
Auto-generate reports/final_report.md and results/metrics_summary.csv from experiment JSON.

Run after: train_baseline, train_cnn, train_transformers, experiment_fusion_ab, run_noise_robustness
"""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUTS_DIR, REPORTS_DIR


def load_json(name: str) -> dict | None:
    p = OUTPUTS_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fmt_auc(v, ci=None) -> str:
    if v is None:
        return "—"
    s = f"{v:.3f}"
    if ci and len(ci) == 2:
        s += f" [{ci[0]:.3f}-{ci[1]:.3f}]"
    return s


def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fusion = load_json("fusion_ab_results.json")
    tr = load_json("transformer_results.json")
    cnn = load_json("cnn_results.json")
    bl = load_json("baseline_results.json")
    nr = load_json("noise_robustness_results.json")

    lines = [
        "# AI-Generated Voice Fraud Detection — Final Report",
        "",
        "## Research Question",
        "",
        "*Can we detect AI-generated speech with sufficient accuracy and robustness under real-world noise conditions, and does combining handcrafted acoustic features with transformer embeddings improve detection?*",
        "",
        "## Findings Summary",
        "",
    ]

    # 1. Fusion
    lines.append("### 1. Feature Fusion")
    lines.append("")
    if fusion and "summary" in fusion:
        lines.append("| Experiment | Test AUC | Test F1 |")
        lines.append("|------------|----------|---------|")
        for r in fusion["summary"]:
            auc_ci = r.get("auc_ci_95")
            auc_str = fmt_auc(r.get("Test AUC"), auc_ci)
            f1_str = f"{r.get('Test F1', 0):.3f}" if r.get("Test F1") is not None else "—"
            lines.append(f"| {r['Experiment']} | {auc_str} | {f1_str} |")
        if "analysis" in fusion:
            lines.append(f"")
            lines.append(f"**Conclusion:** {fusion['analysis'].get('conclusion', '—')}")
    else:
        lines.append("| Experiment | Test AUC |")
        lines.append("|------------|----------|")
        lines.append("| HF Embeddings Only | — |")
        lines.append("| Acoustic Features Only | — |")
        lines.append("| Fusion (HF + Acoustic) | — |")
        lines.append("")
        lines.append("**Conclusion:** Run `python scripts/experiment_fusion_ab.py`")
    lines.append("")
    lines.append("")

    # 2. Model comparison (baseline first, then transformers, then fusion)
    lines.append("### 2. Model Comparison (Clean, Speaker-Disjoint)")
    lines.append("")
    lines.append("| Model | Test AUC | Test F1 |")
    lines.append("|-------|----------|---------|")

    model_rows = []

    def add_model(name, data):
        if data is None:
            return
        auc = data.get("auc_roc")
        f1 = data.get("f1")
        ci = data.get("auc_ci_95")
        auc_str = fmt_auc(auc, ci)
        f1_str = f"{f1:.3f}" if f1 is not None else "—"
        lines.append(f"| {name} | {auc_str} | {f1_str} |")
        model_rows.append((name, auc, f1))

    # Baseline: CNN only
    add_model("Baseline (CNN)", cnn)
    if tr:
        for k, v in tr.items():
            add_model(k, v)
    if bl:
        add_model("RF (optional)", bl.get("random_forest"))
        add_model("LR (optional)", bl.get("logistic_regression"))
    if fusion and "analysis" in fusion:
        a = fusion["analysis"]
        add_model("Fusion", {"auc_roc": a.get("test_auc_fusion"), "f1": None})

    # Write metrics table CSV to results/
    csv_path = OUTPUTS_DIR / "metrics_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Test_AUC", "Test_F1"])
        for name, auc, f1 in model_rows:
            w.writerow([name, f"{auc:.4f}" if auc is not None else "", f"{f1:.4f}" if f1 is not None else ""])
    print(f"Saved {csv_path}")

    lines.append("")
    lines.append("")

    # 3. Noise robustness
    lines.append("### 3. Noise Robustness (0 dB SNR)")
    lines.append("")
    if nr:
        lines.append("| Model | White | Pink | Compression |")
        lines.append("|-------|-------|------|-------------|")
        for model_name, model_data in nr.items():
            row = [model_name]
            for noise in ["white", "pink", "compression"]:
                auc = model_data.get(noise, {}).get("0dB", {}).get("auc")
                row.append(fmt_auc(auc))
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("| Model | White | Pink | Compression |")
        lines.append("|-------|-------|------|-------------|")
        lines.append("| — | — | — | — |")
        lines.append("")
        lines.append("Run `python scripts/run_noise_robustness.py`")
    lines.append("")
    lines.append("")

    # 4. Recommendations
    best_auc = 0.0
    best_name = "—"
    if tr:
        for k, v in tr.items():
            if v.get("auc_roc", 0) > best_auc:
                best_auc = v["auc_roc"]
                best_name = k
    if cnn and cnn.get("auc_roc", 0) > best_auc:
        best_auc = cnn["auc_roc"]
        best_name = "CNN"
    if bl:
        rf_auc = bl.get("random_forest", {}).get("auc_roc", 0)
        if rf_auc > best_auc:
            best_auc = rf_auc
            best_name = "RF"
    if fusion and "analysis" in fusion:
        fa = fusion["analysis"].get("test_auc_fusion", 0)
        if fa > best_auc:
            best_auc = fa
            best_name = "Fusion"

    lines.append("### 4. Recommendations")
    lines.append("")
    lines.append(f"- **Best model (clean):** {best_name}")
    lines.append("- **Deployment:** Latency, robustness trade-offs; use `python scripts/predict.py` for inference")
    lines.append("- **Limitations:** Dataset coverage, TTS diversity, language")
    lines.append("")
    lines.append("")

    # 5. Future work
    lines.append("### 5. Future Work")
    lines.append("")
    lines.append("- Multilingual evaluation")
    lines.append("- Adversarial robustness testing")
    lines.append("- Online/streaming detection")
    lines.append("")

    out_path = REPORTS_DIR / "final_report.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
