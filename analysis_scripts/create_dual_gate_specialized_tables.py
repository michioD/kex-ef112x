#!/usr/bin/env python3
"""
analysis_scripts/create_dual_gate_specialized_tables.py

Recreates Table VII (Table 7) and Table VIII (Table 8) from the thesis report:
- Table VII: Decision Performance for Dual-Gate (Specialized)
- Table VIII: Correct Offloads Breakdown Dual-Gate (Specialized)

Data source:
  results/hilf_results_dual_gate_specialized_fp_fn.csv
"""

import argparse
import glob
import os
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CSV_DUAL_SPEC = RESULTS_DIR / "hilf_results_dual_gate_specialized_fp_fn.csv"


def ensure_results_exist(csv_path: Path):
    if not csv_path.exists():
        sys.path.insert(0, str(REPO_ROOT))
        import simulation_scripts.run_hilf_dual_gate_specialized_cost as hilf_dual_gate_special

        image_paths = sorted(glob.glob(str(REPO_ROOT / "datasets/coco_images/val2017/*.jpg")))[0:5000]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        hilf_dual_gate_special.run_hierarchical_inference_simulation(image_paths, output_csv=str(csv_path))


def generate_tables(csv_path: Path = CSV_DUAL_SPEC) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_results_exist(csv_path)
    frame = pd.read_csv(csv_path)

    scenarios = [
        ("G_S Reject", False, True),
        ("G_W Reject", True, False),
        ("Both Reject", False, False),
        ("Both Accept", True, True),
    ]

    matrix_rows = []
    breakdown_rows = []

    for label, ssm_dec, weakest_dec in scenarios:
        subset = frame[
            (frame["SSM decision"] == ssm_dec) &
            (frame["Weakest Link decision"] == weakest_dec)
        ]

        if label == "Both Accept":
            correct_accept = int((subset["Y_t"] == 0).sum())
            incorrect_accept = int((subset["Y_t"] == 1).sum())
            correct_offload = 0
            incorrect_offload = 0
        else:
            correct_accept = 0
            incorrect_accept = 0
            correct_offload = int((subset["Y_t"] == 1).sum())
            incorrect_offload = int((subset["Y_t"] == 0).sum())

        matrix_rows.append({
            "Rejection Scenario": label,
            "Correct Accept": correct_accept,
            "Incorrect Accept": incorrect_accept,
            "Correct Offload": correct_offload,
            "Incorrect Offload": incorrect_offload,
        })

        if label != "Both Accept":
            correct_offloads_sub = subset[subset["Y_t"] == 1]
            fn_count = int((correct_offloads_sub["FN"] > 0).sum())
            fp_count = int((correct_offloads_sub["FP"] > 0).sum())
            misc_count = int((correct_offloads_sub["Misclassified"] > 0).sum())
            total_samples = len(correct_offloads_sub)

            breakdown_rows.append({
                "Rejection Scenario": label,
                "FN": fn_count,
                "FP": fp_count,
                "Misclass.": misc_count,
                "Total Samples": total_samples,
            })

    table7 = pd.DataFrame(matrix_rows)
    table8 = pd.DataFrame(breakdown_rows)
    return table7, table8


def format_table7_latex(df: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Decision Performance for Dual-Gate (Specialized)}",
        "\\begin{tabular}{@{}l@{\\hskip 3pt}c@{\\hskip 3pt}c@{\\hskip 3pt}c@{\\hskip 3pt}c@{}}",
        "\\toprule",
        " & \\textbf{Correct Accept} & \\textbf{Incorrect Accept} & \\textbf{Correct Offload} & \\textbf{Incorrect Offload} \\\\ ",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        label = f"${row['Rejection Scenario']}$" if "_" in row["Rejection Scenario"] else row["Rejection Scenario"]
        lines.append(f"{label:<13} & {row['Correct Accept']:<3} & {row['Incorrect Accept']:<3} & {row['Correct Offload']:<4} & {row['Incorrect Offload']} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:dual_spec_matrix}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def format_table8_latex(df: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Correct Offloads Breakdown Dual-Gate (Specialized)}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "\\textbf{Rejection Scenario} & \\textbf{FN} & \\textbf{FP} & \\textbf{Misclass.} & \\textbf{Total Samples} \\\\ ",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        label = f"${row['Rejection Scenario']}$" if "_" in row["Rejection Scenario"] else row["Rejection Scenario"]
        lines.append(f"{label:<13} & {row['FN']:<5} & {row['FP']:<5} & {row['Misclass.']:<5} & {row['Total Samples']} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:rejection_failure_sources}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Recreate Table VII (7) and Table VIII (8) for Dual-Gate (Specialized).")
    parser.add_argument("--csv", type=Path, default=CSV_DUAL_SPEC, help="Path to hilf_results_dual_gate_specialized_fp_fn.csv")
    parser.add_argument("--latex", action="store_true", help="Output tables in LaTeX format.")
    args = parser.parse_args()

    table7, table8 = generate_tables(args.csv)

    if args.latex:
        print(format_table7_latex(table7))
        print("\n")
        print(format_table8_latex(table8))
    else:
        sep = "=" * 78
        print(f"\n{sep}")
        print("Table VII (Table 7): Decision Performance for Dual-Gate (Specialized)".center(78))
        print(f"{sep}\n")
        print(table7.to_string(index=False))

        print(f"\n\n{sep}")
        print("Table VIII (Table 8): Correct Offloads Breakdown Dual-Gate (Specialized)".center(78))
        print(f"{sep}\n")
        print(table8.to_string(index=False))
        print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
