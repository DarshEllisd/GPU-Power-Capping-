# GPU Power Capping via Clock Tuning

This project is for finding GPU clock settings that reduce energy with minimal runtime loss.

Primary target:
- Keep runtime loss around 2-5%.
- Maximize energy savings under that runtime constraint.

## Research Questions

1. What is the relation between frequency, runtime, and energy?
2. Which workload types benefit from frequency reduction?

## Setup (Windows + NVIDIA)

Prerequisites:
- NVIDIA GPU with driver tools (`nvidia-smi` available in PATH)
- Administrator privileges (required for clock locking on many systems)
- Python 3.10+

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

## Step 1: Sweep Workload Set Across Clock Caps

Run the workload sweep script:

```powershell
python scripts/run_workload_set_sweeps.py
```

Useful examples:

```powershell
# Run compute workloads only
python scripts/run_workload_set_sweeps.py --max-memory-files 0

# Run memory workloads only
python scripts/run_workload_set_sweeps.py --max-compute-files 0

# Control repeat count
python scripts/run_workload_set_sweeps.py --repeats-per-cap 3 --baseline-repeats 3
```

Outputs are written to `data/raw/`:
- Per-workload capped runs under `data/raw/compute/` and `data/raw/memory/`
- Per-workload baseline runs under `data/raw/compute/baseline/` and `data/raw/memory/baseline/`
- Timestamped nvidia-smi logs under each workload's `logs/` directory

## Step 2: Analyze Workload-Set Results

Run the analysis script:

```powershell
python scripts/analyze_workload_set_results.py
```

Useful examples:

```powershell
# Analyze only compute workloads
python scripts/analyze_workload_set_results.py --categories compute

# Analyze only memory workloads
python scripts/analyze_workload_set_results.py --categories memory

# Use a custom slowdown cap for recommendation
python scripts/analyze_workload_set_results.py --max-slowdown-pct 3
```

Outputs are written to `data/processed/`:
- Per-workload analysis CSVs under `data/processed/compute/` and `data/processed/memory/`
- Combined summary at `data/processed/workload_set_analysis_summary.csv`
- Combined points table at `data/processed/workload_set_all_points.csv`

## Step 3: Interpret and Compare Tradeoffs

Use the processed CSVs to compare:
- slowdown vs. baseline
- energy savings vs. baseline
- EDP / normalized EDP
- recommended clock under the configured slowdown budget

## Notes

- On some GPUs, supported lock clocks are discrete. The script auto-snaps to nearest supported clock.
- For stable measurements, run each clock multiple times (`repeats >= 3`).
- Keep the system awake during long sweeps to avoid transient CUDA/driver errors after sleep-resume.

## Archived

The following components are currently out of scope and kept only for historical reference while the modeling pipeline is being redesigned:

- `src/train_policy.py`
- `src/rl_api.py`
- `src/recommend_caps.py`
- `src/train_rl_policy.py`
- `src/infer_rl_cap.py`
- `src/live_rl_controller.py`

Current active workflow uses only:

- `scripts/run_workload_set_sweeps.py`
- `scripts/analyze_workload_set_results.py`
