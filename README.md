# GPU Power Capping via Clock Tuning

This project is for finding GPU clock settings that reduce energy with minimal runtime loss.

Primary target:
- Keep runtime loss around 2-5%.
- Maximize energy savings under that runtime constraint.

## Research Questions

1. What is the relation between frequency, runtime, and energy?
2. Which workload types benefit from frequency reduction?
3. Can we automate clock selection using ML?

## Setup (Windows + NVIDIA)

Prerequisites:
- NVIDIA GPU with driver tools (`nvidia-smi` available in PATH)
- Administrator privileges (required for clock locking on many systems)
- Python 3.10+

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

## Step 1: Sweep GPU Core Clocks

Edit [configs/sweep.example.yaml](configs/sweep.example.yaml) for your benchmark command and clock range.

Prebuilt benchmark configs (ported from workload logic in `gpu_rl_project_v5/workload.py`):
- [configs/sweep.compute.yaml](configs/sweep.compute.yaml)
- [configs/sweep.memory.yaml](configs/sweep.memory.yaml)
- [configs/sweep.mixed.yaml](configs/sweep.mixed.yaml)

Run sweep (Python-only):

```bash
python scripts/run_sweep.py --config configs/sweep.example.yaml
```

Examples:

```bash
python scripts/run_sweep.py --config configs/sweep.compute.yaml
python scripts/run_sweep.py --config configs/sweep.memory.yaml
python scripts/run_sweep.py --config configs/sweep.mixed.yaml
```

This creates CSV files in `data/raw/<job_name>/` containing:
- timestamped power logs
- per-run summary (runtime, average power, estimated energy)

Benchmark implementation used by these configs:
- [benchmarks/pytorch_workload_bench.py](benchmarks/pytorch_workload_bench.py)

## Step 2: Analyze Tradeoffs

```powershell
python src/analyze_results.py --input data/raw/<job_name>/summary.csv --output data/processed/<job_name>_analysis.csv
```

Outputs include:
- runtime slowdown vs baseline (%)
- energy savings vs baseline (%)
- EDP and normalized EDP
- suggested clock under slowdown cap (default 5%)

## Step 3: Workload-Type Study

Collect runs for different workloads and sizes. For each job, store a profile in:
- `data/features/jobs.csv` (feature table)
- `data/features/labels.csv` (best clock label from analysis)

Use features like:
- SM utilization
- memory controller utilization
- DRAM throughput
- arithmetic intensity proxy
- achieved occupancy

## Step 4: Automation Baseline

Train a baseline model:

```powershell
python src/train_policy.py --features data/features/jobs.csv --labels data/features/labels.csv --output models/clock_policy.pkl
```

This gives a simple policy model you can later replace with RL.

## Step 5: RL-Ready Control API (Budget-Aware)

The project now includes a production-oriented RL API skeleton:
- [src/rl_api.py](src/rl_api.py)

It provides:
- app-agnostic state vector builder (`build_state_vector`)
- constrained reward function (`compute_reward`)
- safety shield for runtime action filtering (`SafetyShield`)
- budget-constrained cap selection (`select_clock_under_budget`)

Generate recommended caps for multiple slowdown budgets from analysis CSVs:

```bash
python src/recommend_caps.py --inputs data/processed/compute_vs_baseline_analysis.csv data/processed/memory_vs_baseline_analysis.csv data/processed/mixed_vs_baseline_analysis.csv --budgets 2,5,10
```

This prints the best cap under each budget using the sweep-derived tradeoffs.

## Step 6: Train RL Policy

Train a single budget-aware RL policy using your processed sweep analysis files:

```bash
python src/train_rl_policy.py --compute data/processed/compute_vs_baseline_analysis.csv --memory data/processed/memory_vs_baseline_analysis.csv --mixed data/processed/mixed_vs_baseline_analysis.csv --budgets 2,5,10 --timesteps 50000 --out-model models/ppo_cap_policy --out-meta models/ppo_cap_policy_meta.json
```

Train with TensorBoard logging:

```bash
python src/train_rl_policy.py --compute data/processed/compute_vs_baseline_analysis.csv --memory data/processed/memory_vs_baseline_analysis.csv --mixed data/processed/mixed_vs_baseline_analysis.csv --budgets 2,5,10 --timesteps 50000 --tb-logdir logs/tensorboard --tb-run-name ppo_cap_v1 --out-model models/ppo_cap_policy --out-meta models/ppo_cap_policy_meta.json
```

Open TensorBoard in another terminal:

```bash
tensorboard --logdir logs/tensorboard
```

## Step 7: Infer Cap from RL + Safety Shield

Example inference for mixed workload with 5% slowdown budget:

```bash
python src/infer_rl_cap.py --model models/ppo_cap_policy --meta models/ppo_cap_policy_meta.json --analysis data/processed/mixed_vs_baseline_analysis.csv --workload mixed --budget 5 --current-clock 2100
```

Output includes:
- proposed clock from RL policy
- safe clock after shield filtering
- expected slowdown/energy-saving for selected cap

## Step 8: Live Controller (No Rollback Mode)

If `mem_util` is not informative on your system, use execution-time proxy from active GPU bursts instead.

Run live controller (no rollback safety logic):

```bash
python src/live_rl_controller.py --model models/ppo_cap_policy --meta models/ppo_cap_policy_meta.json --gpu-index 0 --budget 5 --interval-sec 1.0 --window-size 8 --max-jump 120 --util-active-threshold 70
```

Optional: if you have a baseline latency proxy value, pass it for slowdown monitoring output:

```bash
python src/live_rl_controller.py --model models/ppo_cap_policy --meta models/ppo_cap_policy_meta.json --gpu-index 0 --budget 5 --baseline-latency-sec 1.25
```

Notes:
- This mode classifies workload online (`compute`/`memory`/`mixed`) using util + burst-duration proxy.
- It does NOT use rollback guards; it only applies jump-smoothing (`--max-jump`) and nearest-clock snapping.

## Notes

- On some GPUs, supported lock clocks are discrete. The script auto-snaps to nearest supported clock.
- For stable measurements, run each clock multiple times (`repeats >= 3`).
- Ignore first-run warmup or include explicit warmup iterations in benchmark command.
