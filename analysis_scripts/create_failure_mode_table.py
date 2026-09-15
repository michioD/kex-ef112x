import argparse
from pathlib import Path
import pandas as pd
import glob
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CSV_NAIVE = RESULTS_DIR / "hilf_results_naive.csv"

# Optional simulation trigger if CSV doesn't exist
if not CSV_NAIVE.exists():
    sys.path.insert(0, str(REPO_ROOT))
    import simulation_scripts.run_hilf_naive as hilf_naive

    image_paths = sorted(glob.glob(str(REPO_ROOT / "datasets/coco_images/val2017/*.jpg")))[0:5000]
    hilf_naive.run_hierarchical_inference_simulation(image_paths, output_csv=str(CSV_NAIVE))


def generate_table(csv_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    naive_path = csv_dir / "hilf_results_naive.csv"
    df = pd.read_csv(naive_path)

    # Leaked errors: incorrectly accepted (Action == 'ACCEPTED' and Y_t == 1)
    ia = df[(df["Action"] == "ACCEPTED") & (df["Y_t"] == 1)]
    # Caught errors: correctly offloaded (Action == 'OFFLOADED' and Y_t == 1)
    co = df[(df["Action"] == "OFFLOADED") & (df["Y_t"] == 1)]

    total_ia = len(ia)
    total_co = len(co)

    fn_ia = int((ia["FN"] > 0).sum())
    fp_ia = int((ia["FP"] > 0).sum())
    mis_ia = int((ia["Misclassified"] > 0).sum())

    fn_co = int((co["FN"] > 0).sum())
    fp_co = int((co["FP"] > 0).sum())
    mis_co = int((co["Misclassified"] > 0).sum())

    rows = [
        {
            "Failure Mode": "False Negatives (FN)",
            "Incorrect Accepts": f"{fn_ia} ({fn_ia / total_ia:.1%})",
            "Correct Offloads": f"{fn_co} ({fn_co / total_co:.1%})",
        },
        {
            "Failure Mode": "False Positives (FP)",
            "Incorrect Accepts": f"{fp_ia} ({fp_ia / total_ia:.1%})",
            "Correct Offloads": f"{fp_co} ({fp_co / total_co:.1%})",
        },
        {
            "Failure Mode": "Misclassifications",
            "Incorrect Accepts": f"{mis_ia} ({mis_ia / total_ia:.1%})",
            "Correct Offloads": f"{mis_co} ({mis_co / total_co:.1%})",
        },
        {
            "Failure Mode": "Total Samples",
            "Incorrect Accepts": f"{total_ia}",
            "Correct Offloads": f"{total_co}",
        },
    ]

    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame) -> str:
    latex_rows = []
    for idx, row in df.iterrows():
        if idx == len(df) - 1:
            latex_rows.append("\\midrule")
            mode = f"\\textbf{{{row['Failure Mode']}}}"
            ia = f"\\textbf{{{row['Incorrect Accepts']}}}"
            co = f"\\textbf{{{row['Correct Offloads']}}}"
        else:
            mode = row["Failure Mode"]
            ia = row["Incorrect Accepts"]
            co = row["Correct Offloads"]
        line = f"{mode:<25} & {ia:<18} & {co:<18} \\\\"
        latex_rows.append(line)

    content = "\n".join(latex_rows)
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\caption{Failure Sources for Leaked and Caught Errors (Naive HIL-F)}\n"
        "\\begin{tabular}{@{}lcc@{}}\n"
        "\\toprule\n"
        "\\textbf{Failure Mode} & \\textbf{Incorrect Accepts} & \\textbf{Correct Offloads} \\\\ \n"
        "\\midrule\n"
        f"{content}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\label{tab:failure_modes}\n"
        "\\end{table}"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Table IV from the thesis report.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Path to results directory containing CSVs.")
    parser.add_argument("--latex", action="store_true", help="Print table in LaTeX format.")
    args = parser.parse_args()

    df = generate_table(csv_dir=args.results_dir)

    if args.latex:
        print(to_latex(df))
    else:
        print("=" * 70)
        print("Table IV: Failure Sources for Leaked and Caught Errors (Naive HIL-F)".center(70))
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)


if __name__ == "__main__":
    main()
