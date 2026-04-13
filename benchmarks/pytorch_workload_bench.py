from __future__ import annotations

import argparse
import time

import torch


def run_compute(size: int, repeats: int) -> None:
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")

    torch.cuda.synchronize()

    for _ in range(repeats):
        _ = torch.matmul(a, b)

    torch.cuda.synchronize()


def run_memory(size: int, repeats: int) -> None:
    for _ in range(repeats):
        a = torch.empty((size, size), device="cuda")
        b = torch.empty_like(a)
        a.fill_(1.0)
        b.copy_(a)
        c = b[:, ::2].contiguous()
        d = c.t().contiguous()
        e = d + 1.0
        _ = e.sum().item()


def run_mixed(size: int, repeats: int) -> None:
    for _ in range(repeats):
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")
        c = torch.matmul(a, b)
        d = c.clone()
        _ = d.sum()
        _ = torch.matmul(c, d)


def run_phased(
    size: int,
    repeats: int,
    compute_phase_min: float,
    memory_phase_min: float,
    phase_cycles: int,
    compute_intensity: int,
) -> None:
    """
    Alternates compute-heavy and memory-heavy phases in minute-scale windows.
    """
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")

    compute_sec = max(compute_phase_min, 1.0 / 60.0) * 60.0
    memory_sec = max(memory_phase_min, 1.0 / 60.0) * 60.0
    cycles = max(int(phase_cycles), 1)

    intensity = max(int(compute_intensity), 1)

    for _ in range(repeats):
        for _ in range(cycles):
            # Compute-bound burst: maximize math per byte by reusing resident tensors.
            t_end = time.perf_counter() + compute_sec
            while time.perf_counter() < t_end:
                for _ in range(intensity):
                    c = torch.matmul(a, b)
                    d = torch.matmul(c, b)
                    a = torch.relu(d)

            # Memory-bound burst: emphasize allocation/copy/transpose traffic.
            t_end = time.perf_counter() + memory_sec
            while time.perf_counter() < t_end:
                x = torch.empty((size, size), device="cuda")
                x.fill_(1.0)
                y = x[:, ::2].contiguous()
                z = y.t().contiguous()
                _ = (z + 1.0).sum().item()


def run_workload(
    kind: str,
    size: int,
    repeats: int,
    compute_phase_min: float = 1.0,
    memory_phase_min: float = 1.0,
    phase_cycles: int = 20,
    compute_intensity: int = 3,
) -> None:
    if kind == "compute":
        run_compute(size=size, repeats=repeats)
        return
    if kind == "memory":
        run_memory(size=size, repeats=repeats)
        return
    if kind == "mixed":
        run_mixed(size=size, repeats=repeats)
        return
    if kind == "phased":
        run_phased(
            size=size,
            repeats=repeats,
            compute_phase_min=compute_phase_min,
            memory_phase_min=memory_phase_min,
            phase_cycles=phase_cycles,
            compute_intensity=compute_intensity,
        )
        return
    raise ValueError(f"Unknown workload kind: {kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["compute", "memory", "mixed", "phased"], required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--compute-phase-min", type=float, default=1.0, help="for kind=phased")
    parser.add_argument("--memory-phase-min", type=float, default=1.0, help="for kind=phased")
    parser.add_argument("--phase-cycles", type=int, default=20, help="for kind=phased")
    parser.add_argument(
        "--compute-intensity",
        type=int,
        default=3,
        help="for kind=phased: GEMM burst multiplier in compute phase",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Warmup loops not counted in reported runtime",
    )
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This benchmark requires a CUDA-capable GPU.")

    args = parse_args()

    for _ in range(max(args.warmup_steps, 0)):
        run_workload(
            kind=args.kind,
            size=args.size,
            repeats=args.repeats,
            compute_phase_min=args.compute_phase_min,
            memory_phase_min=args.memory_phase_min,
            phase_cycles=args.phase_cycles,
            compute_intensity=args.compute_intensity,
        )
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.steps):
        run_workload(
            kind=args.kind,
            size=args.size,
            repeats=args.repeats,
            compute_phase_min=args.compute_phase_min,
            memory_phase_min=args.memory_phase_min,
            phase_cycles=args.phase_cycles,
            compute_intensity=args.compute_intensity,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    runtime = end - start
    print(f"kind={args.kind}")
    print(f"size={args.size}")
    print(f"repeats={args.repeats}")
    print(f"steps={args.steps}")
    print(f"runtime_s={runtime:.6f}")


if __name__ == "__main__":
    main()
