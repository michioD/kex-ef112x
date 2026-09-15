import argparse
import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulation_scripts.run_hilf_naive as hilf_naive
import simulation_scripts.run_hilf_dual_gate_global_cost as hilf_dual_gate_global
import simulation_scripts.run_hilf_dual_gate_specialized_cost as hilf_dual_gate_special


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

CSV_NAIVE = RESULTS_DIR / "hilf_results_naive.csv"
CSV_DUAL_GLOBAL = RESULTS_DIR / "hilf_results_dual_gate_same_yt.csv"
CSV_DUAL_SPEC = RESULTS_DIR / "hilf_results_dual_gate_specialized_fp_fn.csv"

image_paths = sorted(glob.glob(str(REPO_ROOT / "datasets/coco_images/val2017/*.jpg")))[0:5000]
if not (RESULTS_DIR / "hilf_results_naive.csv").exists(): 
    hilf_naive.run_hierarchical_inference_simulation(image_paths, output_csv=str(CSV_NAIVE))
if not (RESULTS_DIR / "hilf_results_dual_gate_same_yt.csv").exists():
    hilf_dual_gate_global.run_hierarchical_inference_simulation(image_paths, output_csv=str(CSV_DUAL_GLOBAL))
if not (RESULTS_DIR / "hilf_results_dual_gate_specialized_fp_fn.csv").exists():
    hilf_dual_gate_special.run_hierarchical_inference_simulation(image_paths, output_csv=str(CSV_DUAL_SPEC))


def compute_policy_metrics(name: str, frame: pd.DataFrame, accept_mask: pd.Series, beta: float = 0.5) -> dict:
    y = frame["Y_t"].astype(int)
    offload_mask = ~accept_mask
    correct_accept = int((accept_mask & (y == 0)).sum())
    incorrect_accept = int((accept_mask & (y == 1)).sum())
    correct_offload = int((offload_mask & (y == 1)).sum())
    incorrect_offload = int((offload_mask & (y == 0)).sum())

    n_samples = len(frame)
    accepted_count = int(accept_mask.sum())
    offloaded_count = n_samples - accepted_count

    dec_acc = (correct_accept + correct_offload) / n_samples
    e2e_acc = (correct_accept + offloaded_count) / n_samples
    local_acc = (correct_accept / accepted_count) if accepted_count > 0 else 0.0
    avg_cost = (incorrect_accept + beta * offloaded_count) / n_samples
    offload_pct = offloaded_count / n_samples

    return {
        "Policy": name,
        "Dec. Acc.": dec_acc,
        "E2E Acc.": e2e_acc,
        "Local Acc.": local_acc,
        "Avg. Cost": avg_cost,
        "Offload %": offload_pct,
    }


def generate_table(csv_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    naive_path = csv_dir / "hilf_results_naive.csv"
    global_path = csv_dir / "hilf_results_dual_gate_same_yt.csv"
    spec_path = csv_dir / "hilf_results_dual_gate_specialized_fp_fn.csv"

    naive_path = "results/hilf_results_naive.csv"
    global_path = "results/hilf_results_dual_gate_same_yt.csv"
    spec_path = "results/hilf_results_dual_gate_specialized_fp_fn.csv"

    df_naive = pd.read_csv(naive_path)
    df_global = pd.read_csv(global_path)
    df_spec = pd.read_csv(spec_path)

    n_samples = len(df_naive)
    rng = np.random.default_rng(42)

    rows = [
        compute_policy_metrics("Full Accept", df_naive, pd.Series(True, index=df_naive.index)),
        compute_policy_metrics("Coin Flip (50%)", df_naive, pd.Series(rng.random(n_samples) < 0.5, index=df_naive.index)),
        compute_policy_metrics("Full Offload", df_naive, pd.Series(False, index=df_naive.index)),
        compute_policy_metrics("Naive HIL-F", df_naive, df_naive["Action"] == "ACCEPTED"),
        compute_policy_metrics("SSM", df_global, df_global["SSM decision"] == True),
        compute_policy_metrics("Dual-Gate (Global)", df_global, df_global["Action"] == "ACCEPTED"),
        compute_policy_metrics("Dual-Gate (Spec.)", df_spec, df_spec["Action"] == "ACCEPTED"),
        compute_policy_metrics("Genie Benchmark", df_naive, df_naive["Y_t"] == 0),
    ]

    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for col in ["Dec. Acc.", "E2E Acc.", "Local Acc.", "Offload %"]:
        formatted[col] = formatted[col].map("{:.1%}".format)
    formatted["Avg. Cost"] = formatted["Avg. Cost"].map("{:.3f}".format)
    return formatted


def to_latex(df_formatted: pd.DataFrame) -> str:
    latex_rows = []
    for idx, row in df_formatted.iterrows():
        line = f"{row['Policy']:<18} & {row['Dec. Acc.']:<7} & {row['E2E Acc.']:<7} & {row['Local Acc.']:<7} & {row['Avg. Cost']:<6} & {row['Offload %']:<7} \\\\"
        if idx in (2, 6):
            line += " \\midrule"
        latex_rows.append(line)

    content = "\n".join(latex_rows)
    return (
        "\\begin{table*}[!t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\caption{Performance Comparison of Different Offloading Policies on COCO Validation Set}\n"
        "\\begin{tabular}{@{}lccccc@{}}\n"
        "\\toprule\n"
        "\\textbf{Policy} & \\textbf{Dec. Acc.} & \\textbf{E2E Acc.} & \\textbf{Local Acc.} & \\textbf{Avg. Cost} & \\textbf{Offload \\%} \\\\ \\midrule\n"
        f"{content}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\label{tab:policy_comparison}\n"
        "\\end{table*}"
    )


def main():
    raw_df = generate_table()
    formatted_df = format_table(raw_df)
    print("=" * 80)
    print("Table II: Performance Comparison of Different Offloading Policies".center(80))
    print("=" * 80)
    print(formatted_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
