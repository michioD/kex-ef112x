# Hierarchical Inference Learning for Edge Offloading

This repository contains the code, thesis source, reference results, and figure-generation material for a study of adaptive object-detection offloading. The experiment compares YOLOv8n (edge model) with YOLOv8x (offloaded model) on the COCO 2017 validation split.

## Repository guide

| Path | Purpose |
|---|---|
| `src/` | HIL-F algorithm, detection metrics, and baseline policies |
| `scripts/` | Dataset caching, evaluation, reproduction, and reporting entry points |
| `results/` | Historical reference CSVs used by the thesis |
| `docs/report-clean/` | LaTeX thesis source and compiled report |
| `notebooks/thesis_replica.ipynb` | Figure/table replica based on the reference results |
| `old-code/` | Archived exploratory work; not part of the supported workflow |
| `artifacts/runs/` | New, self-contained evaluation runs (ignored by Git) |

## Quick start

Use a clean virtual environment and the tested package versions:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make data
make cache
make reproduce
```

`make data` downloads COCO 2017 validation images and annotations. `make cache` runs YOLOv8n and YOLOv8x once per image; it is compute- and storage-intensive.

The `.pt` model files (`yolov8n.pt` and `yolov8x.pt`) are automatically downloaded by the `ultralytics` Python library if they are not found locally. This occurs during precomputation in [`scripts/precompute.py`](scripts/precompute.py) and [`scripts/precompute_raw.py`](scripts/precompute_raw.py) when initializing the models:

```python
s_ml = YOLO('yolov8n.pt')
l_ml = YOLO('yolov8x.pt')
```

## System & Storage Requirements

Running the full preprocessing, inference caching, and evaluation pipeline requires adequate disk space and memory:

- **Storage Requirements (~25 GB total):**
  - `datasets/`: ~1.5 GB (COCO 2017 validation images and annotations)
  - `cache_yolo_results/`: ~11 GB (standard precomputed YOLOv8n and YOLOv8x inference tensors)
  - `cache_yolo_results_raw/`: ~11 GB (raw pre-NMS detection predictions)
  - Model weights (`.pt` files) & run artifacts: ~500 MB - 1 GB
- **Hardware & Environment:**
  - **Memory (RAM):** 16 GB+ recommended (32 GB+ recommended for running unbatched raw feature clustering/caching smoothly)
  - **Compute:** Apple Silicon (M-series with MPS support) or an NVIDIA GPU with CUDA recommended for timely precomputation across all 5,000 validation images; CPU execution is functional but significantly slower.
  - **Python:** Python 3.9+ (tested on Python 3.9.6 / macOS 15.7.2)

## Reproducible experiments

Every new run is isolated and includes input settings, environment details, and SHA-256 checksums:

```bash
python3 -m scripts.reproduce --policies naive ssm --limit 5000 --seed 42
```

The command creates `artifacts/runs/<UTC timestamp>/` containing per-policy CSVs, `summary.csv`, `summary.md`, and `manifest.json`. It never overwrites the historical files in `results/`. To reproduce an exact folder name, pass `--run-id NAME`; the command deliberately refuses to overwrite it.

For a fast smoke test after caching, use:

```bash
python3 -m scripts.reproduce --policies naive --limit 10 --seed 42 --run-id smoke-test
```

The random policy decisions are seeded. With the same cached predictions, software environment, input ordering, policy, limit, and seed, the resulting CSV is deterministic. Fresh YOLO inference can vary across hardware and backends; retain the generated cache when exact numerical replication is required.

## Supported policies

`naive`, `dual_gate`, `dual_gate_same_yt`, `ssm`, and `dual_gate_detailed` are available. Outputs share a standard core schema (`Action`, `Y_t`, FP/FN/misclassification fields); dual-gate outputs add per-gate decisions.

## Thesis materials

The report source is [docs/report-clean/main.tex](docs/report-clean/main.tex). The notebook loads the reference data in `results/`; run it from the repository root so relative paths resolve correctly. The legacy scripts in `old-code/` are retained for traceability but are not maintained as a reproducible pipeline.

## Data and provenance

The supported dataset is COCO 2017 validation (`val2017`, 5,000 images). The downloader uses the official COCO host. Reference result CSVs are preserved as the data behind the submitted thesis; new experiments belong in `artifacts/runs/`, each with its own manifest.
