from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from types import ModuleType

import torch


def numeric_prefix(stem: str) -> tuple[int, str]:
    head = stem.split("_", 1)[0]
    if head.isdigit():
        return int(head), stem
    return 10**9, stem


def discover_workloads(folder: Path, max_files: int) -> list[Path]:
    files = sorted((p for p in folder.glob("*.py") if p.is_file()), key=lambda p: numeric_prefix(p.stem))
    if max_files > 0:
        return files[:max_files]
    return files


def load_module(path: Path) -> ModuleType:
    module_name = f"workload_{path.stem}_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load workload module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model_and_inputs(module: ModuleType) -> tuple[torch.nn.Module, list[object]]:
    if not hasattr(module, "Model"):
        raise ValueError("Missing Model class")
    if not hasattr(module, "get_inputs"):
        raise ValueError("Missing get_inputs()")
    if not hasattr(module, "get_init_inputs"):
        raise ValueError("Missing get_init_inputs()")

    init_inputs = list(module.get_init_inputs())
    model = module.Model(*init_inputs).to("cuda").eval()

    raw_inputs = list(module.get_inputs())
    inputs: list[object] = []
    for item in raw_inputs:
        if isinstance(item, torch.Tensor):
            inputs.append(item.to("cuda", non_blocking=True))
        else:
            inputs.append(item)
    return model, inputs


def run_workload_module(path: Path, model_repeats: int) -> None:
    module = load_module(path)
    model, inputs = build_model_and_inputs(module)

    repeats = max(int(model_repeats), 1)
    with torch.no_grad():
        for _ in range(repeats):
            out = model(*inputs)
            # Touch output references so each forward pass is retained as real work.
            if isinstance(out, (list, tuple)):
                _ = out[0] if out else None
            else:
                _ = out



def run_phase(workloads: list[Path], model_repeats: int) -> None:
    for path in workloads:
        run_workload_module(path, model_repeats=model_repeats)



def run_sequence(
    compute_workloads: list[Path],
    memory_workloads: list[Path],
    pattern: str,
    phase_cycles: int,
    model_repeats: int,
) -> None:
    tokens = [x.strip().lower() for x in pattern.split("-") if x.strip()]
    if not tokens:
        raise ValueError("Empty --pattern; expected values like c-m-c")

    for token in tokens:
        if token not in {"c", "m", "compute", "memory"}:
            raise ValueError(f"Unsupported pattern token: {token}")

    cycles = max(int(phase_cycles), 1)
    for _ in range(cycles):
        for token in tokens:
            if token in {"c", "compute"}:
                run_phase(compute_workloads, model_repeats=model_repeats)
            else:
                run_phase(memory_workloads, model_repeats=model_repeats)
            torch.cuda.synchronize()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-dir", default="workloads/compute_bound")
    parser.add_argument("--memory-dir", default="workloads/memory_bound")
    parser.add_argument("--max-compute-files", type=int, default=0, help="0 means all files")
    parser.add_argument("--max-memory-files", type=int, default=0, help="0 means all files")
    parser.add_argument("--pattern", default="c-m-c", help="Phase order, e.g. c-m-c")
    parser.add_argument("--phase-cycles", type=int, default=1, help="How many times to repeat the phase pattern")
    parser.add_argument("--model-repeats", type=int, default=1, help="Forward passes per workload per phase")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    return parser.parse_args()



def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This benchmark requires a CUDA-capable GPU.")

    args = parse_args()
    compute_dir = Path(args.compute_dir)
    memory_dir = Path(args.memory_dir)

    if not compute_dir.exists():
        raise FileNotFoundError(f"Compute workload directory not found: {compute_dir}")
    if not memory_dir.exists():
        raise FileNotFoundError(f"Memory workload directory not found: {memory_dir}")

    compute_workloads = discover_workloads(compute_dir, max_files=args.max_compute_files)
    memory_workloads = discover_workloads(memory_dir, max_files=args.max_memory_files)

    if not compute_workloads:
        raise ValueError(f"No compute workloads discovered in {compute_dir}")
    if not memory_workloads:
        raise ValueError(f"No memory workloads discovered in {memory_dir}")

    print(f"kind=workload_sequence")
    print(f"compute_count={len(compute_workloads)}")
    print(f"memory_count={len(memory_workloads)}")
    print(f"pattern={args.pattern}")
    print(f"phase_cycles={args.phase_cycles}")
    print(f"model_repeats={args.model_repeats}")

    for _ in range(max(args.warmup_steps, 0)):
        run_sequence(
            compute_workloads=compute_workloads,
            memory_workloads=memory_workloads,
            pattern=args.pattern,
            phase_cycles=args.phase_cycles,
            model_repeats=args.model_repeats,
        )
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(max(args.steps, 1)):
        run_sequence(
            compute_workloads=compute_workloads,
            memory_workloads=memory_workloads,
            pattern=args.pattern,
            phase_cycles=args.phase_cycles,
            model_repeats=args.model_repeats,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    runtime = end - start
    print(f"runtime_s={runtime:.6f}")


if __name__ == "__main__":
    main()
