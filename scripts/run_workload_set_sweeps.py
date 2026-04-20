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


def run_checked(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


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
    query = (
        "timestamp,power.draw,clocks.gr,clocks.sm,clocks.mem,temperature.gpu,"
        "utilization.gpu,utilization.memory,memory.used,memory.total,"
        "pstate,pcie.link.gen.gpucurrent,pcie.link.width.current"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,nounits",
        "-i",
        str(gpu_index),
        "-lms",
        str(interval_ms),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def measure_run_energy(
    log_path: Path, runtime_sec: float
) -> tuple[float, float, float, float, float, float, float, float, float]:
    if not log_path.exists():
        return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan

    powers: list[float] = []
    clocks: list[float] = []
    temps: list[float] = []
    gpu_utils: list[float] = []
    mem_bw_utils: list[float] = []
    vram_used_mib: list[float] = []
    vram_alloc_pct: list[float] = []
    with log_path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            parts = text.split(",")
            if len(parts) < 10:
                continue
            try:
                powers.append(float(parts[1].strip()))
                clocks.append(float(parts[2].strip()))
                temps.append(float(parts[5].strip()))
                gpu_util = float(parts[6].strip())
                mem_bw_util = float(parts[7].strip())
                used_mib = float(parts[8].strip())
                total_mib = float(parts[9].strip())

                gpu_utils.append(gpu_util)
                mem_bw_utils.append(mem_bw_util)
                vram_used_mib.append(used_mib)
                if total_mib > 0:
                    vram_alloc_pct.append((used_mib / total_mib) * 100.0)
            except ValueError:
                continue

    if not powers:
        return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan

    avg_power = sum(powers) / len(powers)
    avg_clock = sum(clocks) / len(clocks) if clocks else math.nan
    peak_clock = max(clocks) if clocks else math.nan
    avg_temp = sum(temps) / len(temps) if temps else math.nan
    avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else math.nan
    avg_mem_bw_util = sum(mem_bw_utils) / len(mem_bw_utils) if mem_bw_utils else math.nan
    avg_vram_used_mib = sum(vram_used_mib) / len(vram_used_mib) if vram_used_mib else math.nan
    avg_vram_alloc_pct = sum(vram_alloc_pct) / len(vram_alloc_pct) if vram_alloc_pct else math.nan
    energy_j = avg_power * runtime_sec
    return (
        avg_power,
        avg_clock,
        peak_clock,
        avg_temp,
        avg_gpu_util,
        avg_mem_bw_util,
        avg_vram_used_mib,
        avg_vram_alloc_pct,
        energy_j,
    )


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
                    "peak_clock_mhz",
                    "avg_temp_c",
                    "avg_gpu_util_pct",
                    "avg_mem_bw_util_pct",
                    "avg_vram_used_mib",
                    "avg_vram_alloc_pct",
                    "runtime_s",
                    "avg_power_w",
                    "energy_j",
                    "log_path",
                ]
            )
        writer.writerow(row)


def discover_workloads(folder: Path, max_files: int) -> list[Path]:
    files = sorted((p for p in folder.glob("*.py") if p.is_file()), key=lambda p: p.name.lower())
    if max_files == 0:
        return []
    if max_files > 0:
        return files[:max_files]
    return files


def run_module_benchmark(module_path: Path, model_repeats: int, steps: int, warmup_steps: int) -> float:
    cmd = [
        sys.executable,
        "benchmarks/workload_module_bench.py",
        "--module",
        str(module_path),
        "--model-repeats",
        str(model_repeats),
        "--steps",
        str(steps),
        "--warmup-steps",
        str(warmup_steps),
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            "Benchmark subprocess failed. "
            f"module={module_path} returncode={exc.returncode}\n"
            f"stdout:\n{stdout or '<empty>'}\n"
            f"stderr:\n{stderr or '<empty>'}"
        ) from exc
    runtime_sec: float | None = None
    for line in result.stdout.splitlines():
        text = line.strip()
        if text.startswith("runtime_s="):
            try:
                runtime_sec = float(text.split("=", 1)[1].strip())
            except ValueError:
                runtime_sec = None
            break
    if runtime_sec is None:
        raise ValueError(
            f"Failed to parse runtime_s from benchmark output for {module_path}:\n{result.stdout}"
        )
    return runtime_sec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--compute-dir", default="workloads/compute_bound")
    parser.add_argument("--memory-dir", default="workloads/memory_bound")
    parser.add_argument("--max-compute-files", type=int, default=-1)
    parser.add_argument("--max-memory-files", type=int, default=-1)
    parser.add_argument("--clock-caps", default="3105,3045,2985,2925,2865,2805,2745,2685,2625,2565,2505,2445,2385,2325,2265,2205,2145,2085,2025,1965,1905,1845,1785,1725,1665,1605,1545,1485,1425,1365,1305,1245,1185,1065,1005,945,885,825,765,705,645,585,525")
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--warmup-seconds", type=int, default=2)
    parser.add_argument("--repeats-per-cap", type=int, default=3)
    parser.add_argument("--baseline-repeats", type=int, default=3)
    parser.add_argument("--bench-model-repeats", type=int, default=1)
    parser.add_argument("--bench-steps", type=int, default=1)
    parser.add_argument("--bench-warmup-steps", type=int, default=1)
    parser.add_argument("--out-root", default="data/raw")
    return parser.parse_args()


def run_category(
    category_name: str,
    workloads: list[Path],
    out_root: Path,
    gpu_index: int,
    caps: list[int],
    supported: list[int],
    sample_interval_ms: int,
    warmup_seconds: int,
    repeats_per_cap: int,
    baseline_repeats: int,
    bench_model_repeats: int,
    bench_steps: int,
    bench_warmup_steps: int,
) -> None:
    category_root = out_root / category_name
    baseline_root = category_root / "baseline"

    for workload_path in workloads:
        workload_name = workload_path.stem
        print(f"\n=== {category_name}/{workload_name} ===")

        sweep_csv = category_root / f"{workload_name}.csv"
        baseline_csv = baseline_root / f"{workload_name}.csv"
        sweep_logs_root = category_root / "logs" / workload_name
        baseline_logs_root = baseline_root / "logs" / workload_name

        # Baseline runs first: no lock, repeated baseline_repeats times.
        print(f"[{workload_name}] baseline repeats={baseline_repeats}")
        reset_clock(gpu_index)
        if warmup_seconds > 0:
            time.sleep(warmup_seconds)

        for rep in range(1, baseline_repeats + 1):
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = f"baseline_r{rep}_{ts}"
            log_path = baseline_logs_root / f"{run_id}_power.csv"

            logger_proc, logger_file = start_power_logger(gpu_index, log_path, sample_interval_ms)
            time.sleep(0.3)
            try:
                runtime_sec = run_module_benchmark(
                    workload_path,
                    model_repeats=bench_model_repeats,
                    steps=bench_steps,
                    warmup_steps=bench_warmup_steps,
                )
            finally:
                stop_power_logger(logger_proc, logger_file)

            (
                avg_power,
                avg_clock,
                peak_clock,
                avg_temp,
                avg_gpu_util,
                avg_mem_bw_util,
                avg_vram_used_mib,
                avg_vram_alloc_pct,
                energy_j,
            ) = measure_run_energy(log_path, runtime_sec)
            append_row(
                baseline_csv,
                [
                    run_id,
                    -1,
                    -1,
                    f"{avg_clock:.4f}",
                    f"{peak_clock:.4f}",
                    f"{avg_temp:.4f}",
                    f"{avg_gpu_util:.4f}",
                    f"{avg_mem_bw_util:.4f}",
                    f"{avg_vram_used_mib:.4f}",
                    f"{avg_vram_alloc_pct:.4f}",
                    f"{runtime_sec:.4f}",
                    f"{avg_power:.4f}",
                    f"{energy_j:.4f}",
                    str(log_path),
                ],
            )
            print(
                f"  baseline rep={rep} gpu_runtime={runtime_sec:.3f}s clock={avg_clock:.1f}MHz "
                f"peakClock={peak_clock:.1f}MHz temp={avg_temp:.1f}C gpuUtil={avg_gpu_util:.1f}% "
                f"memBw={avg_mem_bw_util:.1f}% vram={avg_vram_used_mib:.1f}MiB ({avg_vram_alloc_pct:.1f}%) "
                f"power={avg_power:.2f}W energy={energy_j:.2f}J"
            )

        # Capped sweep runs: for each cap, run repeats_per_cap times.
        for cap in caps:
            applied = select_nearest_clock(cap, supported)
            print(f"[{workload_name}] cap target={cap} applied={applied} repeats={repeats_per_cap}")
            set_clock(gpu_index, applied)

            if warmup_seconds > 0:
                time.sleep(warmup_seconds)

            for rep in range(1, repeats_per_cap + 1):
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                run_id = f"clk{applied}_r{rep}_{ts}"
                log_path = sweep_logs_root / f"{run_id}_power.csv"

                logger_proc, logger_file = start_power_logger(gpu_index, log_path, sample_interval_ms)
                time.sleep(0.3)
                try:
                    runtime_sec = run_module_benchmark(
                        workload_path,
                        model_repeats=bench_model_repeats,
                        steps=bench_steps,
                        warmup_steps=bench_warmup_steps,
                    )
                finally:
                    stop_power_logger(logger_proc, logger_file)

                (
                    avg_power,
                    avg_clock,
                    peak_clock,
                    avg_temp,
                    avg_gpu_util,
                    avg_mem_bw_util,
                    avg_vram_used_mib,
                    avg_vram_alloc_pct,
                    energy_j,
                ) = measure_run_energy(log_path, runtime_sec)
                append_row(
                    sweep_csv,
                    [
                        run_id,
                        cap,
                        applied,
                        f"{avg_clock:.4f}",
                        f"{peak_clock:.4f}",
                        f"{avg_temp:.4f}",
                        f"{avg_gpu_util:.4f}",
                        f"{avg_mem_bw_util:.4f}",
                        f"{avg_vram_used_mib:.4f}",
                        f"{avg_vram_alloc_pct:.4f}",
                        f"{runtime_sec:.4f}",
                        f"{avg_power:.4f}",
                        f"{energy_j:.4f}",
                        str(log_path),
                    ],
                )
                print(
                    f"  cap={applied} rep={rep} gpu_runtime={runtime_sec:.3f}s clock={avg_clock:.1f}MHz "
                    f"peakClock={peak_clock:.1f}MHz temp={avg_temp:.1f}C gpuUtil={avg_gpu_util:.1f}% "
                    f"memBw={avg_mem_bw_util:.1f}% vram={avg_vram_used_mib:.1f}MiB ({avg_vram_alloc_pct:.1f}%) "
                    f"power={avg_power:.2f}W energy={energy_j:.2f}J"
                )

        reset_clock(gpu_index)


def main() -> int:
    args = parse_args()

    compute_dir = Path(args.compute_dir)
    memory_dir = Path(args.memory_dir)
    out_root = Path(args.out_root)

    if not compute_dir.exists():
        raise FileNotFoundError(f"Compute workload directory not found: {compute_dir}")
    if not memory_dir.exists():
        raise FileNotFoundError(f"Memory workload directory not found: {memory_dir}")

    compute_workloads = discover_workloads(compute_dir, max_files=args.max_compute_files)
    memory_workloads = discover_workloads(memory_dir, max_files=args.max_memory_files)

    if args.max_compute_files != 0 and not compute_workloads:
        raise ValueError(f"No workload files in {compute_dir}")
    if args.max_memory_files != 0 and not memory_workloads:
        raise ValueError(f"No workload files in {memory_dir}")

    caps = [int(x.strip()) for x in args.clock_caps.split(",") if x.strip()]
    if not caps:
        raise ValueError("No caps found in --clock-caps")

    supported = get_supported_clocks(args.gpu_index)
    if supported:
        print(f"Supported clocks: {', '.join(str(c) for c in supported)}")

    try:
        if compute_workloads:
            run_category(
                category_name="compute",
                workloads=compute_workloads,
                out_root=out_root,
                gpu_index=args.gpu_index,
                caps=caps,
                supported=supported,
                sample_interval_ms=args.sample_interval_ms,
                warmup_seconds=args.warmup_seconds,
                repeats_per_cap=args.repeats_per_cap,
                baseline_repeats=args.baseline_repeats,
                bench_model_repeats=args.bench_model_repeats,
                bench_steps=args.bench_steps,
                bench_warmup_steps=args.bench_warmup_steps,
            )

        if memory_workloads:
            run_category(
                category_name="memory",
                workloads=memory_workloads,
                out_root=out_root,
                gpu_index=args.gpu_index,
                caps=caps,
                supported=supported,
                sample_interval_ms=args.sample_interval_ms,
                warmup_seconds=args.warmup_seconds,
                repeats_per_cap=args.repeats_per_cap,
                baseline_repeats=args.baseline_repeats,
                bench_model_repeats=args.bench_model_repeats,
                bench_steps=args.bench_steps,
                bench_warmup_steps=args.bench_warmup_steps,
            )
    finally:
        try:
            reset_clock(args.gpu_index)
        except Exception as exc:
            print(f"Warning: failed to reset clocks: {exc}")

    print(f"Done. Outputs under: {out_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
