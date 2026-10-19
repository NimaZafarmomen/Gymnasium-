"""Tabular Q-learning for Gymnasium CliffWalking, with reproducible experiments.

No reinforcement-learning library implements the agent. Gymnasium supplies
the environment only. See README.md for the complete experimental protocol.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    name: str = "baseline"
    alpha: float = 0.5
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    episodes: int = 5000
    max_steps: int = 500
    noise: float = 0.0
    checkpoint_every: int = 50


class ActionNoise(gym.Wrapper):
    """With probability p, execute one of the THREE DIFFERENT valid actions.

    The intended action remains the action being learned. Noise is sampled
    independently of epsilon-greedy exploration. All four directions are valid,
    including movements against boundaries (which leave the position unchanged).
    """
    def __init__(self, env: gym.Env, p: float = 0.0):
        super().__init__(env)
        if not 0 <= p <= 1:
            raise ValueError("Noise probability must be in [0, 1].")
        self.p = p
        self.noise_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 917]))
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        intended = int(action)
        executed = intended
        if self.noise_rng.random() < self.p:
            # Offset 1, 2 or 3 modulo 4 never returns the intended action.
            executed = (intended + int(self.noise_rng.integers(1, 4))) % 4
        obs, reward, terminated, truncated, info = self.env.step(executed)
        info = dict(info, intended_action=intended, executed_action=executed,
                    action_replaced=(executed != intended))
        return obs, reward, terminated, truncated, info


def make_env(config: Config, render_mode=None):
    return ActionNoise(gym.make("CliffWalking-v1", is_slippery=False,
                               max_episode_steps=config.max_steps,
                               render_mode=render_mode), config.noise)


def epsilon_at(config: Config, episode_zero_based: int) -> float:
    return max(config.epsilon_min,
               config.epsilon_start * config.epsilon_decay ** episode_zero_based)


def choose_action(q: np.ndarray, state: int, epsilon: float,
                  rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    values = q[state]
    # Random tie-breaking prevents a permanent bias to the first action.
    maxima = np.flatnonzero(values == values.max())
    return int(rng.choice(maxima))


def q_update(q, state, action, reward, next_state, terminated, alpha, gamma):
    # Truncation ends this rollout but does NOT suppress bootstrapping.
    target = reward if terminated else reward + gamma * float(q[next_state].max())
    q[state, action] += alpha * (target - q[state, action])


def evaluate(q: np.ndarray, config: Config, n_episodes: int, seed_base: int,
             train_seed: int, phase: str) -> pd.DataFrame:
    """Frozen greedy policy; deterministic argmax ties; no Q updates."""
    env = make_env(config)
    rows = []
    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed_base + episode)
        total, falls = 0.0, 0
        for step in range(1, config.max_steps + 1):
            action = int(np.argmax(q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            falls += int(reward == -100)
            if terminated or truncated:
                break
        rows.append(dict(config=config.name, train_seed=train_seed, phase=phase,
                         eval_noise=config.noise, episode=episode + 1,
                         env_seed=seed_base + episode, reward=total, steps=step,
                         cliff_falls=falls, success=int(terminated),
                         truncated=int(truncated and not terminated)))
    env.close()
    return pd.DataFrame(rows)


def train_one(task):
    config, seed, root = task
    root = Path(root)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    env = make_env(config)
    q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    rows, checkpoints = [], []
    for episode in range(config.episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        epsilon = epsilon_at(config, episode)
        total, falls = 0.0, 0
        for step in range(1, config.max_steps + 1):
            action = choose_action(q, state, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            q_update(q, state, action, reward, next_state, terminated,
                     config.alpha, config.gamma)
            total += reward
            falls += int(reward == -100)
            state = next_state
            if terminated or truncated:
                break
        rows.append(dict(config=config.name, train_seed=seed, episode=episode + 1,
                         epsilon=epsilon, reward=total, steps=step,
                         cliff_falls=falls, success=int(terminated),
                         truncated=int(truncated and not terminated)))
        if (episode + 1) % config.checkpoint_every == 0:
            # Independent validation RNG: never consumes the training RNG.
            count = 1 if config.noise == 0 else 30
            result = evaluate(q, config, count, 100000 + seed * 1000,
                              seed, "checkpoint")
            checkpoints.append(dict(config=config.name, train_seed=seed,
                                    episode=episode + 1,
                                    reward=result.reward.mean(),
                                    success=result.success.mean(),
                                    steps=result.steps.mean()))
    env.close()
    run_dir = root / "runs" / config.name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / "training.csv", index=False)
    pd.DataFrame(checkpoints).to_csv(run_dir / "checkpoints.csv", index=False)
    np.save(run_dir / "q_table.npy", q)
    validation = evaluate(q, config, 100, 200000 + seed * 1000, seed, "validation")
    test = evaluate(q, config, 100, 300000 + seed * 1000, seed, "test")
    validation.to_csv(run_dir / "validation.csv", index=False)
    test.to_csv(run_dir / "test.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    return config.name, seed


def deterministic_configs(base: Config):
    return [base, replace(base, name="alpha_0.1", alpha=0.1),
            replace(base, name="alpha_0.9", alpha=0.9),
            replace(base, name="gamma_0.5", gamma=0.5),
            replace(base, name="gamma_0.9", gamma=0.9),
            replace(base, name="decay_0.95", epsilon_decay=0.95),
            replace(base, name="decay_0.999", epsilon_decay=0.999)]


def collect(root, names, filename):
    return pd.concat([pd.read_csv(p) for name in names
                      for p in sorted((Path(root) / "runs" / name).glob(f"seed_*/{filename}.csv"))],
                     ignore_index=True)


def run_tasks(configs, seeds, root, workers):
    tasks = [(c, s, str(root)) for c in configs for s in seeds]
    if workers == 1:
        for task in tasks:
            print("Finished", *train_one(task), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(train_one, tasks):
                print("Finished", *result, flush=True)


def run_early_diagnostics(configs, seeds, root, workers):
    """Replay the first 200 episodes at finer validation resolution.

    This diagnostic is excluded from configuration selection. Equality with
    the original training prefix checks that validation cannot alter learning.
    """
    diagnostic_root = Path(root) / "diagnostics"
    short = [replace(c, episodes=min(200, c.episodes), checkpoint_every=5) for c in configs]
    run_tasks(short, seeds, diagnostic_root, workers)
    for config in configs:
        for seed in seeds:
            relative = Path("runs") / config.name / f"seed_{seed}" / "training.csv"
            full = pd.read_csv(Path(root) / relative).iloc[:min(200, config.episodes)]
            replay = pd.read_csv(diagnostic_root / relative)
            pd.testing.assert_frame_equal(full.reset_index(drop=True), replay)
    collect(diagnostic_root, [c.name for c in configs], "checkpoints").to_csv(
        Path(root) / "early_checkpoints.csv", index=False)
    print("Early diagnostic prefixes exactly match original training.", flush=True)


def run_all(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    base = Config(episodes=args.episodes)
    configs = deterministic_configs(base)
    seeds = list(range(args.seeds))
    protocol = dict(environment="CliffWalking-v1", gymnasium=gym.__version__,
                    numpy=np.__version__, python=platform.python_version(),
                    train_seeds=seeds, configs=[asdict(c) for c in configs],
                    selection_rule=["highest final validation mean reward",
                                    "highest mean checkpoint reward in episodes 50 to 1000",
                                    "highest overall training mean reward",
                                    "lexicographic configuration name"],
                    evaluation_episodes_per_seed=100,
                    noise_checkpoint_episodes=30,
                    validation_seed_base=200000, test_seed_base=300000,
                    robustness_seed_base=400000)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    run_tasks(configs, seeds, root, args.workers)
    names = [c.name for c in configs]
    validation = collect(root, names, "validation")
    checkpoints = collect(root, names, "checkpoints")
    training = collect(root, names, "training")
    # Selection uses validation and training only; final test is never ranked.
    ranking = validation.groupby("config").reward.mean().rename("validation_reward").to_frame()
    ranking["early_checkpoint_reward"] = checkpoints[checkpoints.episode <= 1000].groupby("config").reward.mean()
    ranking["training_reward"] = training.groupby("config").reward.mean()
    ranking = ranking.reset_index().sort_values(
        ["validation_reward", "early_checkpoint_reward", "training_reward", "config"],
        ascending=[False, False, False, True])
    ranking.to_csv(root / "selection_ranking.csv", index=False)
    best = next(c for c in configs if c.name == ranking.iloc[0]["config"])
    (root / "best_config.json").write_text(json.dumps(asdict(best), indent=2))
    print("SELECTED", asdict(best), flush=True)
    noisy = [replace(best, name=f"noise_{p:.2f}", noise=p) for p in [0.10, 0.25]]
    run_tasks(noisy, seeds, root, args.workers)
    protocol["noise_configs"] = [asdict(c) for c in noisy]
    protocol["best_config"] = asdict(best)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    transfer = []
    for p in [0.0, 0.10, 0.25]:
        for seed in seeds:
            q = np.load(root / "runs" / best.name / f"seed_{seed}" / "q_table.npy")
            result = evaluate(q, replace(best, noise=p), 100,
                              400000 + seed * 1000, seed, "zero_noise_transfer")
            transfer.append(result)
            if p:
                name = f"noise_{p:.2f}"
                q_noise = np.load(root / "runs" / name / f"seed_{seed}" / "q_table.npy")
                trained = evaluate(q_noise, replace(best, name=name, noise=p), 100,
                                   400000 + seed * 1000, seed, "matched_noise")
                transfer.append(trained)
    pd.concat(transfer, ignore_index=True).to_csv(root / "robustness.csv", index=False)
    for stem in ["training", "checkpoints", "validation", "test"]:
        collect(root, names + [c.name for c in noisy], stem).to_csv(root / f"{stem}.csv", index=False)
    run_early_diagnostics(configs, seeds, root, args.workers)
    print("All experiments complete.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    run_all(parser.parse_args())


if __name__ == "__main__":
    main()
