from __future__ import annotations

import argparse
import csv
import math
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


@dataclass
class RunResult:
    workload: str
    mode: str
    repeat: int
    runtime_s: float
    avg_power_w: float
    avg_clock_mhz: float
    energy_j: float
    power_log: str
    controller_log: str


def run_checked(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def run_benchmark(command: str) -> float:
    start = time.perf_counter()
    subprocess.run(command, check=True, shell=True)
    end = time.perf_counter()
    return end - start


def reset_clock(gpu_index: int) -> None:
    run_checked(["nvidia-smi", "-i", str(gpu_index), "-rgc"])


def start_power_logger(gpu_index: int, log_path: Path, interval_ms: int) -> tuple[subprocess.Popen[str], object]:
    query = "timestamp,power.draw,clocks.gr,utilization.gpu,utilization.memory"
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
        "-lms",
        str(interval_ms),
    ]
    out_file = log_path.open("w", encoding="ascii", newline="")
    process = subprocess.Popen(cmd, stdout=out_file, stderr=subprocess.DEVNULL, text=True)
    return process, out_file


def stop_power_logger(process: subprocess.Popen[str], out_file: object) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    out_file.close()


def measure_run_energy(log_path: Path, runtime_sec: float) -> tuple[float, float, float]:
    if not log_path.exists():
        return math.nan, math.nan, math.nan

    powers: list[float] = []
    clocks: list[float] = []
    with log_path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            parts = text.split(",")
            if len(parts) < 3:
                continue
            try:
                powers.append(float(parts[1].strip()))
                clocks.append(float(parts[2].strip()))
            except ValueError:
                continue

    if not powers:
        return math.nan, math.nan, math.nan

    avg_power = sum(powers) / len(powers)
    avg_clock = sum(clocks) / len(clocks) if clocks else math.nan
    energy_j = avg_power * runtime_sec
    return avg_power, avg_clock, energy_j


def extract_workload_name(command: str, fallback: str) -> str:
    marker = "--kind "
    if marker not in command:
        return fallback
    tail = command.split(marker, 1)[1].strip()
    if not tail:
        return fallback
    return tail.split()[0]


def start_live_controller(
    python_exe: str,
    gpu_index: int,
    model: str,
    meta: str,
    analysis_compute: str,
    analysis_memory: str,
    analysis_mixed: str,
    budget: float,
    guard_band_pct: float,
    interval_sec: float,
    max_jump: int,
    window_size: int,
    util_active_threshold: float,
    controller_log_path: Path,
) -> tuple[subprocess.Popen[str], object]:
    cmd = [
        python_exe,
        "-u",
        "src/live_rl_controller.py",
        "--model",
        model,
        "--meta",
        meta,
        "--analysis-compute",
        analysis_compute,
        "--analysis-memory",
        analysis_memory,
        "--analysis-mixed",
        analysis_mixed,
        "--gpu-index",
        str(gpu_index),
        "--budget",
        str(budget),
        "--guard-band-pct",
        str(guard_band_pct),
        "--interval-sec",
        str(interval_sec),
        "--window-size",
        str(window_size),
        "--max-jump",
        str(max_jump),
        "--util-active-threshold",
        str(util_active_threshold),
    ]

    out_file = controller_log_path.open("w", encoding="ascii", newline="")
    process = subprocess.Popen(cmd, stdout=out_file, stderr=subprocess.STDOUT, text=True)
    return process, out_file


def stop_live_controller(process: subprocess.Popen[str], out_file: object) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=4)
    out_file.close()


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/sweep.compute.yaml",
            "configs/sweep.memory.yaml",
            "configs/sweep.mixed.yaml",
            "configs/sweep.phased.yaml",
        ],
        help="YAML workload configs to evaluate",
    )
    parser.add_argument("--model", default="models/ppo_cap_policy_v3")
    parser.add_argument("--meta", default="models/ppo_cap_policy_v3_meta.json")
    parser.add_argument("--analysis-compute", default="data/processed/compute_vs_baseline_analysis.csv")
    parser.add_argument("--analysis-memory", default="data/processed/memory_vs_baseline_analysis.csv")
    parser.add_argument("--analysis-mixed", default="data/processed/mixed_vs_baseline_analysis.csv")
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--guard-band-pct", type=float, default=0.5)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--max-jump", type=int, default=120)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--util-active-threshold", type=float, default=70.0)
    parser.add_argument("--controller-lead-sec", type=float, default=1.2)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-seconds", type=int, default=-1)
    parser.add_argument("--out-dir", default="data/raw/rl_eval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    detailed_csv = out_dir / f"rl_vs_baseline_detailed_{timestamp}.csv"

    python_exe = sys.executable
    results: list[RunResult] = []

    for cfg_path_text in args.configs:
        cfg_path = Path(cfg_path_text)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        gpu_index = int(cfg.get("gpu", {}).get("index", 0))
        benchmark_cfg = cfg.get("benchmark", {})
        command = str(benchmark_cfg.get("command", "")).strip()
        if not command:
            raise ValueError(f"benchmark.command is empty in {cfg_path}")

        job_name = str(benchmark_cfg.get("job_name", cfg_path.stem))
        warmup_seconds = int(benchmark_cfg.get("warmup_seconds", 0))
        if args.warmup_seconds >= 0:
            warmup_seconds = args.warmup_seconds

        workload = extract_workload_name(command, fallback=job_name)

        print(f"\n=== Workload: {workload} ({job_name}) ===")
        print(f"Command: {command}")

        for mode in ("baseline", "rl"):
            print(f"\n[{workload}] Mode={mode} | repeats={args.repeats}")
            for rep in range(1, args.repeats + 1):
                run_tag = f"{workload}_{mode}_r{rep}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
                power_log = out_dir / f"{run_tag}_power.csv"
                controller_log = out_dir / f"{run_tag}_controller.log"

                reset_clock(gpu_index)
                if warmup_seconds > 0:
                    time.sleep(warmup_seconds)

                ctrl_proc = None
                ctrl_file = None
                logger_proc = None
                logger_file = None

                try:
                    if mode == "rl":
                        ctrl_proc, ctrl_file = start_live_controller(
                            python_exe=python_exe,
                            gpu_index=gpu_index,
                            model=args.model,
                            meta=args.meta,
                            analysis_compute=args.analysis_compute,
                            analysis_memory=args.analysis_memory,
                            analysis_mixed=args.analysis_mixed,
                            budget=args.budget,
                            guard_band_pct=args.guard_band_pct,
                            interval_sec=args.interval_sec,
                            max_jump=args.max_jump,
                            window_size=args.window_size,
                            util_active_threshold=args.util_active_threshold,
                            controller_log_path=controller_log,
                        )
                        time.sleep(max(args.controller_lead_sec, 0.2))

                    logger_proc, logger_file = start_power_logger(
                        gpu_index=gpu_index,
                        log_path=power_log,
                        interval_ms=args.sample_interval_ms,
                    )
                    time.sleep(0.3)

                    runtime_s = run_benchmark(command)

                finally:
                    if logger_proc is not None and logger_file is not None:
                        stop_power_logger(logger_proc, logger_file)
                    if ctrl_proc is not None and ctrl_file is not None:
                        stop_live_controller(ctrl_proc, ctrl_file)
                    reset_clock(gpu_index)

                avg_power_w, avg_clock_mhz, energy_j = measure_run_energy(power_log, runtime_s)
                print(
                    f"  rep={rep} runtime={runtime_s:.3f}s power={avg_power_w:.2f}W clock={avg_clock_mhz:.1f}MHz energy={energy_j:.2f}J"
                )

                results.append(
                    RunResult(
                        workload=workload,
                        mode=mode,
                        repeat=rep,
                        runtime_s=runtime_s,
                        avg_power_w=avg_power_w,
                        avg_clock_mhz=avg_clock_mhz,
                        energy_j=energy_j,
                        power_log=str(power_log),
                        controller_log=str(controller_log) if mode == "rl" else "",
                    )
                )

    with detailed_csv.open("w", encoding="ascii", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "workload",
                "mode",
                "repeat",
                "runtime_s",
                "avg_power_w",
                "avg_clock_mhz",
                "energy_j",
                "power_log",
                "controller_log",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.workload,
                    r.mode,
                    r.repeat,
                    f"{r.runtime_s:.6f}",
                    f"{r.avg_power_w:.6f}",
                    f"{r.avg_clock_mhz:.1f}",
                    f"{r.energy_j:.6f}",
                    r.power_log,
                    r.controller_log,
                ]
            )

    workloads = sorted({r.workload for r in results})
    print("\n=== RL vs Baseline Summary ===")
    for workload in workloads:
        base = [r for r in results if r.workload == workload and r.mode == "baseline"]
        rl = [r for r in results if r.workload == workload and r.mode == "rl"]
        if not base or not rl:
            continue

        b_runtime = mean([x.runtime_s for x in base])
        b_power = mean([x.avg_power_w for x in base])
        b_clock = mean([x.avg_clock_mhz for x in base])
        b_energy = mean([x.energy_j for x in base])

        r_runtime = mean([x.runtime_s for x in rl])
        r_power = mean([x.avg_power_w for x in rl])
        r_clock = mean([x.avg_clock_mhz for x in rl])
        r_energy = mean([x.energy_j for x in rl])

        slowdown_pct = ((r_runtime / b_runtime) - 1.0) * 100.0 if b_runtime > 1e-9 else math.nan
        power_saving_pct = (1.0 - (r_power / b_power)) * 100.0 if b_power > 1e-9 else math.nan
        energy_saving_pct = (1.0 - (r_energy / b_energy)) * 100.0 if b_energy > 1e-9 else math.nan

        print(
            " | ".join(
                [
                    f"workload={workload}",
                    f"baseline_runtime={b_runtime:.3f}s",
                    f"rl_runtime={r_runtime:.3f}s",
                    f"slowdown={slowdown_pct:.2f}%",
                    f"baseline_clock={b_clock:.1f}MHz",
                    f"rl_clock={r_clock:.1f}MHz",
                    f"baseline_energy={b_energy:.2f}J",
                    f"rl_energy={r_energy:.2f}J",
                    f"energy_saving={energy_saving_pct:.2f}%",
                    f"power_saving={power_saving_pct:.2f}%",
                ]
            )
        )

    print(f"\nDetailed results saved: {detailed_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
