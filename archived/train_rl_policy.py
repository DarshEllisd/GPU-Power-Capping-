from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from archived.rl_api import RewardConfig, compute_reward


class SweepRLEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        tables: dict[str, pd.DataFrame],
        budgets: list[float],
        episode_len: int = 24,
    ) -> None:
        super().__init__()
        if not tables:
            raise ValueError("At least one workload table is required")

        self.workload_names = sorted(tables.keys())
        self.tables = tables
        self.budgets = budgets
        self.episode_len = int(episode_len)

        all_clocks = set()
        for df in self.tables.values():
            all_clocks.update(int(c) for c in df["clock_applied_mhz"].tolist())
        self.clock_values = sorted(all_clocks)
        self.clock_to_idx = {c: i for i, c in enumerate(self.clock_values)}

        self.action_space = gym.spaces.Discrete(len(self.clock_values))

        # [onehot(workloads), norm_budget, norm_current_clock]
        self.obs_dim = len(self.workload_names) + 2
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.reward_cfg = RewardConfig()
        self.step_idx = 0
        self.current_budget = 5.0
        self.current_workload = self.workload_names[0]
        self.current_clock = self.clock_values[-1]
        self._reward_template = self.reward_cfg

    def _lookup(self, workload: str, clock: int) -> tuple[float, float]:
        df = self.tables[workload]
        row = df[df["clock_applied_mhz"] == clock]
        if row.empty:
            nearest_idx = int((df["clock_applied_mhz"] - clock).abs().idxmin())
            row = df.loc[[nearest_idx]]
        r = row.iloc[0]
        return float(r["slowdown_pct"]), float(r["energy_saving_pct"])

    def _obs(self) -> np.ndarray:
        onehot = np.zeros(len(self.workload_names), dtype=np.float32)
        onehot[self.workload_names.index(self.current_workload)] = 1.0
        min_clock = float(self.clock_values[0])
        max_clock = float(self.clock_values[-1])
        norm_clock = (float(self.current_clock) - min_clock) / max(max_clock - min_clock, 1.0)
        norm_budget = float(self.current_budget) / 30.0
        return np.concatenate([onehot, np.asarray([norm_budget, norm_clock], dtype=np.float32)])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.current_workload = self.workload_names[int(self.np_random.integers(0, len(self.workload_names)))]
        self.current_budget = float(self.budgets[int(self.np_random.integers(0, len(self.budgets)))])
        # START FROM RANDOM CLOCK (not always max) to force policy to learn state-dependent decisions
        self.current_clock = int(self.clock_values[int(self.np_random.integers(0, len(self.clock_values)))])
        self.reward_cfg = RewardConfig(
            slowdown_budget_pct=self.current_budget,
            slowdown_penalty=self._reward_template.slowdown_penalty,
            jitter_penalty=self._reward_template.jitter_penalty,
            energy_scale=self._reward_template.energy_scale,
            violation_scale=self._reward_template.violation_scale,
            jump_scale_mhz=self._reward_template.jump_scale_mhz,
        )
        return self._obs(), {}

    def step(self, action: int):
        proposed_clock = int(self.clock_values[int(action)])
        slowdown_pct, energy_saving_pct = self._lookup(self.current_workload, proposed_clock)
        jump = proposed_clock - self.current_clock

        reward = compute_reward(
            energy_saving_pct=energy_saving_pct,
            slowdown_pct=slowdown_pct,
            clock_jump_mhz=jump,
            cfg=self.reward_cfg,
        )

        self.current_clock = proposed_clock
        self.step_idx += 1

        terminated = self.step_idx >= self.episode_len
        truncated = False

        info = {
            "workload": self.current_workload,
            "budget": self.current_budget,
            "clock": proposed_clock,
            "slowdown_pct": slowdown_pct,
            "energy_saving_pct": energy_saving_pct,
        }
        return self._obs(), float(reward), terminated, truncated, info


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"clock_applied_mhz", "slowdown_pct", "energy_saving_pct"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.sort_values("clock_applied_mhz").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", required=True, help="compute analysis csv")
    parser.add_argument("--memory", required=True, help="memory analysis csv")
    parser.add_argument("--mixed", required=True, help="mixed analysis csv")
    parser.add_argument("--budgets", default="2,5,10", help="comma-separated slowdown budgets")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--episode-len", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--slowdown-penalty", type=float, default=1.0)
    parser.add_argument("--jitter-penalty", type=float, default=0.05)
    parser.add_argument("--energy-scale", type=float, default=50.0)
    parser.add_argument("--violation-scale", type=float, default=10.0)
    parser.add_argument("--jump-scale-mhz", type=float, default=120.0)
    parser.add_argument("--tb-logdir", default="logs/tensorboard")
    parser.add_argument("--tb-run-name", default="ppo_cap")
    parser.add_argument("--out-model", default="models/ppo_cap_policy")
    parser.add_argument("--out-meta", default="models/ppo_cap_policy_meta.json")
    return parser.parse_args()


def parse_budgets(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No budgets parsed")
    return vals


def main() -> None:
    args = parse_args()
    tables = {
        "compute": load_table(Path(args.compute)),
        "memory": load_table(Path(args.memory)),
        "mixed": load_table(Path(args.mixed)),
    }
    budgets = parse_budgets(args.budgets)

    env = SweepRLEnv(tables=tables, budgets=budgets, episode_len=args.episode_len)
    check_env(env, warn=True)

    env.reward_cfg = RewardConfig(
        slowdown_budget_pct=5.0,
        slowdown_penalty=args.slowdown_penalty,
        jitter_penalty=args.jitter_penalty,
        energy_scale=args.energy_scale,
        violation_scale=args.violation_scale,
        jump_scale_mhz=args.jump_scale_mhz,
    )
    env._reward_template = env.reward_cfg

    model = PPO(
        policy="MlpPolicy",
        env=env,
        device="cpu",
        tensorboard_log=args.tb_logdir if args.tb_logdir else None,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, tb_log_name=args.tb_run_name)

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_model))

    meta = {
        "workloads": env.workload_names,
        "clock_values": env.clock_values,
        "budgets": budgets,
        "obs_dim": env.obs_dim,
    }
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved RL model: {out_model}")
    print(f"Saved metadata: {out_meta}")


if __name__ == "__main__":
    main()
