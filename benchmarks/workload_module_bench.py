from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from types import ModuleType

import torch


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


def run_model(module_path: Path, model_repeats: int) -> None:
    module = load_module(module_path)
    model, inputs = build_model_and_inputs(module)

    repeats = max(int(model_repeats), 1)
    with torch.no_grad():
        for _ in range(repeats):
            out = model(*inputs)
            if isinstance(out, (list, tuple)):
                _ = out[0] if out else None
            else:
                _ = out



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="Path to workload module .py file")
    parser.add_argument("--model-repeats", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    return parser.parse_args()



def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This benchmark requires a CUDA-capable GPU.")

    args = parse_args()
    module_path = Path(args.module)
    if not module_path.exists():
        raise FileNotFoundError(f"Workload module not found: {module_path}")

    for _ in range(max(args.warmup_steps, 0)):
        run_model(module_path, model_repeats=args.model_repeats)
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(max(args.steps, 1)):
        run_model(module_path, model_repeats=args.model_repeats)
    torch.cuda.synchronize()
    end = time.perf_counter()

    runtime = end - start
    print(f"kind=single_workload_module")
    print(f"module={module_path}")
    print(f"model_repeats={args.model_repeats}")
    print(f"steps={args.steps}")
    print(f"runtime_s={runtime:.6f}")


if __name__ == "__main__":
    main()
