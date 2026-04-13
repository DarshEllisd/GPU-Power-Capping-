from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml


def run_checked(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def run_benchmark(command: str) -> float:
    start = time.perf_counter()
    subprocess.run(command, check=True, shell=True)
    end = time.perf_counter()
    return end - start


def get_supported_clocks(gpu_index: int) -> list[int]:
    result = run_checked(
        [
            "nvidia-smi",
            "--query-supported-clocks=graphics",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_index),
        ]
    )
    clocks: list[int] = []
    for raw in result.stdout.splitlines():
        text = raw.strip()
        if text.isdigit():
            clocks.append(int(text))
    return sorted(set(clocks), reverse=True)


def select_nearest_clock(target: int, supported: Iterable[int]) -> int:
    supported_list = list(supported)
    if not supported_list:
        return target
    return min(supported_list, key=lambda c: abs(c - target))


def set_clock(gpu_index: int, clock_mhz: int) -> None:
    run_checked(["nvidia-smi", "-i", str(gpu_index), "-lgc", f"{clock_mhz},{clock_mhz}"])


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
    process = subprocess.Popen(
        cmd,
        stdout=out_file,
        stderr=subprocess.DEVNULL,
        text=True,
    )
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


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    return cfg


def append_row(summary_path: Path, row: list[object]) -> None:
    is_new = not summary_path.exists()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="ascii", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                [
                    "run_id",
                    "clock_target_mhz",
                    "clock_applied_mhz",
                        "avg_clock_mhz",
                    "runtime_s",
                    "avg_power_w",
                    "energy_j",
                    "log_path",
                ]
            )
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep.example.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    gpu_cfg = cfg.get("gpu", {})
    bench_cfg = cfg.get("benchmark", {})
    out_cfg = cfg.get("output", {})

    gpu_index = int(gpu_cfg.get("index", 0))
    targets = [int(x) for x in gpu_cfg.get("clocks_mhz", [])]
    interval_ms = int(gpu_cfg.get("sample_interval_ms", 100))

    job_name = str(bench_cfg.get("job_name", "job"))
    bench_cmd = str(bench_cfg.get("command", ""))
    warmup_sec = int(bench_cfg.get("warmup_seconds", 0))
    repeats = int(bench_cfg.get("repeats", 1))
    out_root = Path(str(out_cfg.get("root_dir", "data/raw")))

    if not bench_cmd:
        raise ValueError("benchmark.command is empty")

    run_dir = out_root / job_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.csv"

    supported = get_supported_clocks(gpu_index)
    if supported:
        print(f"Supported clocks: {', '.join(str(c) for c in supported)}")

    try:
        if not targets:
            print("Running baseline (no clock lock, natural GPU max)...")
            if warmup_sec > 0:
                time.sleep(warmup_sec)

            for rep in range(1, repeats + 1):
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                run_id = f"baseline_r{rep}_{ts}"
                log_path = run_dir / f"{run_id}_power.csv"

                logger_proc, logger_file = start_power_logger(gpu_index, log_path, interval_ms)
                time.sleep(0.3)

                runtime_sec = math.nan
                try:
                    runtime_sec = run_benchmark(bench_cmd)
                finally:
                    stop_power_logger(logger_proc, logger_file)

                avg_power, avg_clock, energy_j = measure_run_energy(log_path, runtime_sec)
                append_row(
                    summary_path,
                    [
                        run_id,
                        -1,
                        -1,
                        f"{avg_clock:.4f}",
                        f"{runtime_sec:.4f}",
                        f"{avg_power:.4f}",
                        f"{energy_j:.4f}",
                        str(log_path),
                    ],
                )
                print(
                    "Run completed: "
                    f"{run_id} | runtime={runtime_sec:.3f}s | avgClock={avg_clock:.1f}MHz | avgPower={avg_power:.2f}W | energy={energy_j:.2f}J"
                )
        else:
            for target_clock in targets:
                applied_clock = select_nearest_clock(target_clock, supported)
                print(f"Setting clock target={target_clock} MHz, applied={applied_clock} MHz")
                set_clock(gpu_index, applied_clock)

                if warmup_sec > 0:
                    time.sleep(warmup_sec)

                for rep in range(1, repeats + 1):
                    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    run_id = f"clk{applied_clock}_r{rep}_{ts}"
                    log_path = run_dir / f"{run_id}_power.csv"

                    logger_proc, logger_file = start_power_logger(gpu_index, log_path, interval_ms)
                    time.sleep(0.3)

                    runtime_sec = math.nan
                    try:
                        runtime_sec = run_benchmark(bench_cmd)
                    finally:
                        stop_power_logger(logger_proc, logger_file)

                    avg_power, avg_clock, energy_j = measure_run_energy(log_path, runtime_sec)
                    append_row(
                        summary_path,
                        [
                            run_id,
                            target_clock,
                            applied_clock,
                            f"{avg_clock:.4f}",
                            f"{runtime_sec:.4f}",
                            f"{avg_power:.4f}",
                            f"{energy_j:.4f}",
                            str(log_path),
                        ],
                    )
                    print(
                        "Run completed: "
                        f"{run_id} | runtime={runtime_sec:.3f}s | avgClock={avg_clock:.1f}MHz | avgPower={avg_power:.2f}W | energy={energy_j:.2f}J"
                    )
    finally:
        print("Resetting GPU clocks")
        try:
            reset_clock(gpu_index)
        except Exception as exc:
            print(f"Warning: failed to reset clocks: {exc}")

    print(f"Done. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
