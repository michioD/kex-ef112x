import argparse
import os
from pathlib import Path
import glob
import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

CSV_NAIVE = RESULTS_DIR / "hilf_results_naive.csv"
CSV_DUAL_GLOBAL = RESULTS_DIR / "hilf_results_dual_gate_same_yt.csv"
CSV_DUAL_SPEC = RESULTS_DIR / "hilf_results_dual_gate_specialized_fp_fn.csv"

# Optional simulation triggers if CSVs don't exist
if not CSV_NAIVE.exists() or not CSV_DUAL_GLOBAL.exists() or not CSV_DUAL_SPEC.exists():
    sys.path.insert(0, str(REPO_ROOT))
    import simulation_scripts.run_hilf_naive as hilf_naive
    import simulation_scripts.run_hilf_dual_gate_global_cost as hilf_dual_gate_global
    import simulation_scripts.run_hilf_dual_gate_specialized_cost as hilf_dual_gate_special

    image_paths = sorted(glob.glob(str(REPO_ROOT / "datasets/coco_images/val2017/*.jpg")))[0:5000]
    if not CSV_NAIVE.exists():
        hilf_naive.run_hierarchical_inference_simulation(image_paths, output_csv=str(CSV_NAIVE))
    if not CSV_DUAL_GLOBAL.exists():
        hilf_dual_gate_global.run_hierarchical_inference_simulation(image_paths, output_csv=str(CSV_DUAL_GLOBAL))
    if not CSV_DUAL_SPEC.exists():
        hilf_dual_gate_special.run_hierarchical_inference_simulation(image_paths, output_csv=str(CSV_DUAL_SPEC))


def compute_decision_breakdown(name: str, frame: pd.DataFrame, accept_mask: pd.Series) -> dict:
    y = frame["Y_t"].astype(int)
    offload_mask = ~accept_mask
    correct_accept = int((accept_mask & (y == 0)).sum())
    incorrect_accept = int((accept_mask & (y == 1)).sum())
    correct_offload = int((offload_mask & (y == 1)).sum())
    incorrect_offload = int((offload_mask & (y == 0)).sum())

    return {
        "Policy": name,
        "Correct Accept": correct_accept,
        "Incorrect Accept": incorrect_accept,
        "Correct Offload": correct_offload,
        "Incorrect Offload": incorrect_offload,
    }


def generate_table(csv_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    naive_path = csv_dir / "hilf_results_naive.csv"
    global_path = csv_dir / "hilf_results_dual_gate_same_yt.csv"
    spec_path = csv_dir / "hilf_results_dual_gate_specialized_fp_fn.csv"

    df_naive = pd.read_csv(naive_path)
    df_global = pd.read_csv(global_path)
    df_spec = pd.read_csv(spec_path)

    rows = [
        compute_decision_breakdown("Naive", df_naive, df_naive["Action"] == "ACCEPTED"),
        compute_decision_breakdown("SSM", df_global, df_global["SSM decision"] == True),
        compute_decision_breakdown("Dual Global", df_global, df_global["Action"] == "ACCEPTED"),
        compute_decision_breakdown("Dual Special", df_spec, df_spec["Action"] == "ACCEPTED"),
    ]

    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame) -> str:
    latex_rows = []
    for _, row in df.iterrows():
        line = f"{row['Policy']:<14} & {row['Correct Accept']:<14} & {row['Incorrect Accept']:<16} & {row['Correct Offload']:<15} & {row['Incorrect Offload']:<17} \\\\"
        latex_rows.append(line)

    content = "\n".join(latex_rows)
    return (
        "\\begin{table*}[!t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\caption{Decision Performance of Different Policies}\n"
        "\\begin{tabular}{@{}l@{\\hskip 4pt}c@{\\hskip 4pt}c@{\\hskip 4pt}c@{\\hskip 4pt}c@{}}\n"
        "\\toprule\n"
        "\\textbf{Policy} & \\textbf{Correct Accept} & \\textbf{Incorrect Accept} & \\textbf{Correct Offload} & \\textbf{Incorrect Offload} \\\\ \n"
        "\\midrule\n"
        f"{content}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\label{tab:all_policy_comparison}\n"
        "\\end{table*}"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Table III from the thesis report.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Path to results directory containing CSVs.")
    parser.add_argument("--latex", action="store_true", help="Print table in LaTeX format.")
    args = parser.parse_args()

    df = generate_table(csv_dir=args.results_dir)

    if args.latex:
        print(to_latex(df))
    else:
        print("=" * 80)
        print("Table III: Decision Performance of Different Policies".center(80))
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)


if __name__ == "__main__":
    main()
