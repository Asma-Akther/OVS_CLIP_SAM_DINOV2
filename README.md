# README — Final Evaluation Notebook

## Overview

This notebook runs the complete experimental evaluation for the paper *"Prompt Sensitivity and Robustness of Grounding DINO, SAM, CLIP, and DINOv2 for Open-Vocabulary Image Segmentation in Real-World Conditions."* It is self-contained: given a fresh Kaggle GPU session, it installs all dependencies, loads all four models (CLIP, SAM, Grounding DINO, DINOv2), loads the RefCOCO-M dataset, and runs every experiment reported in the paper, producing the raw CSV result files used to build the paper's tables and figures.

The notebook is designed to survive interruption. Kaggle's GPU sessions are capped at 12 hours; the full evaluation takes roughly 14 hours end to end. Every experiment is broken into small, independently-saved chunks, so the notebook can be stopped and re-run at any point without losing completed work or recomputing anything already finished.

## Requirements

- Kaggle notebook with GPU accelerator enabled (tested on T4)
- Internet access enabled (for package installs, model downloads, and dataset download)
- No manual setup beyond running cells top to bottom — all installs are handled in Cell 1

## Structure

| Section | Purpose |
|---|---|
| **1. Setup** | Installs dependencies; loads CLIP, SAM, Grounding DINO, DINOv2; loads RefCOCO-M; sets a fixed random seed for reproducibility |
| **1a. Resumable Chunk Runner** | A generic helper (`run_chunked`) used by every experiment below. Splits work into fixed-size image chunks, saves each chunk's results to its own CSV immediately, and skips any chunk whose result file already exists on a re-run |
| **2. Canonical CLIP+SAM Pipeline** | The diagnostic baseline: CLIP dense-heatmap localization + SAM segmentation |
| **3. Canonical Grounding DINO+SAM Pipeline** | The primary pipeline: Grounding DINO localization + SAM segmentation, including typo-tolerant left/right disambiguation |
| **4. DINOv2 (Both Tested Roles)** | Role A: DINOv2-refined CLIP localization. Role B: DINOv2 re-scoring of SAM's candidate masks |
| **5. Corruption Functions** | Gaussian blur, Gaussian noise, and brightness reduction, at both a fixed "moderate" severity and a light/moderate/severe sweep |
| **6. Sanity Check** | Runs both pipelines once on a known case to confirm the environment loaded correctly before committing to long runs |
| **Priority 1-4** | The four supporting experiments (DINOv2-under-corruption, CLIP localization diagnostic, severity sweep, DINOv2 clean re-scoring), run in priority order so the most important new evidence completes first even if the session is interrupted |
| **Main Comparison** | The full clean-vs-corrupted comparison between CLIP+SAM and Grounding DINO+SAM across the complete 1,190-image RefCOCO-M validation set. This is the most expensive section and is placed last so an interruption never costs the higher-priority results above it |
| **Final Summary** | Reads and lists whatever result files currently exist on disk. Safe to run at any time, including mid-run, to check progress |

## How to Run

**Fresh session:** Run all cells top to bottom via *Save Version -> Save & Run All (Commit)*. Do not run it interactively for the full evaluation. Commit executes independently of your browser connection and survives disconnects, which an interactive session does not.

**Resuming after an interruption (e.g., the 12-hour session limit):**
1. Attach the previous (interrupted) version's output as an input dataset via *Add Input -> Notebook Output*.
2. Confirm the mount path under `/kaggle/input/` (a one-off directory listing may be needed to find the exact path; this can be done with a simple `os.walk` over `/kaggle/input`).
3. Ensure the checkpoint-restore step near the top of Setup copies that prior output's `results/` folder into the working `./results/` directory before any experiment section runs.
4. Re-run *Save & Run All (Commit)*. Every already-completed chunk will be detected and skipped automatically; only genuinely unfinished work will be recomputed.

> Note: this copy of the notebook includes a leftover directory-listing cell at the very top, used once during a resume to locate the correct `/kaggle/input/` path. It is not required for a fresh run and can be safely deleted.

## Output

Results are written to `./results/`, organized by experiment:
- `priority1_dinov2_corruption/`, `priority2_clip_diagnostic/`, `priority3_severity_sweep/`, `priority4_dinov2_clean/`: per-chunk checkpoints and final consolidated CSVs for each priority experiment
- `optional_main_experiment/`: per-condition chunk checkpoints and `{condition}_ALL.csv` files (clean, blur, noise, brightness) for the main comparison
- Consolidated top-level CSVs: `dinov2_corruption_final.csv`, `clip_localization_diagnostic.csv`, `severity_sweep_final.csv`, `dinov2_rescore_results.csv`

These are the files used directly to build the paper's Results and Error Analysis tables and figures.

## Runtime

The full evaluation takes approximately **14 hours of GPU compute**, typically requiring two Kaggle sessions due to the 12-hour session cap. Individual experiments run much faster in isolation (roughly 10 minutes to 1.5 hours each); the main comparison over the full 1,190-image dataset across four conditions is the dominant cost.

## Reproducibility

A fixed random seed (`SEED = 42`) is set for NumPy and PyTorch at the start of Setup, so Gaussian noise corruption and any other randomized steps are deterministic across runs and sessions.
