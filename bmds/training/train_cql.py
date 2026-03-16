import numpy as np
from pathlib import Path
from typing import Optional

from bmds.config import (
    CQL_ALPHA, BATCH_SIZE, ACTOR_LR, CRITIC_LR,
    MODELS_DIR, DATA_PROCESSED_DIR,
)
from bmds.training.dataset_builder import DatasetBuilder

REWARD_CLIP = (-50.0, 60.0)

def _save_model_and_scaler(algo, model_path: Path):
    import json
    algo.save_model(str(model_path))
    scaler_info = {}
    try:
        sc = algo.scaler
        if sc is not None and hasattr(sc, '_mean') and sc._mean is not None:
            scaler_info['obs_mean'] = sc._mean.tolist()
            scaler_info['obs_std']  = sc._std.tolist()
    except Exception:
        pass
    try:
        rs = algo.reward_scaler
        if rs is not None and hasattr(rs, '_mean') and rs._mean is not None:
            scaler_info['reward_mean'] = float(rs._mean)
            scaler_info['reward_std']  = float(rs._std)
    except Exception:
        pass
    if scaler_info:
        scaler_path = Path(model_path).with_suffix('.scaler.json')
        with open(scaler_path, 'w') as f:
            json.dump(scaler_info, f, indent=2)

class EarlyStopSignal(Exception):
    pass

_EVAL_MOVEMENTS = [
    ((100, 100), (800, 500)),
    ((960, 100), (200, 800)),
    ((100, 540), (1800, 540)),
    ((960, 50),  (960, 1000)),
    ((200, 200), (400, 300)),
    ((1500, 800),(300, 200)),
    ((500, 500), (1400, 500)),
    ((960, 540), (1600, 200)),
]

def _eval_reach_rate(algo, env, mapper, n_episodes: int = 8) -> float:
    movements = _EVAL_MOVEMENTS[:n_episodes]
    reached = 0
    for (start, end) in movements:
        desk_start = mapper.screen_to_desk(*start)
        desk_end   = mapper.screen_to_desk(*end)
        obs, _ = env.reset(start_pos=desk_start, target_pos=desk_end)
        done = False
        step = 0
        terminated = False
        while not done and step < 500:
            action = algo.predict(np.expand_dims(obs, 0))[0]
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
        if terminated:
            reached += 1
    return reached / len(movements)

class EarlyStopMonitor:

    def __init__(self, eval_env, eval_mapper, save_path,
                 n_eval: int = 8,
                 target_reach_rate: float = 0.625,
                 patience: int = 4,
                 min_evals_before_patience: int = 4,
                 eval_every: int = 1):
        self.eval_env   = eval_env
        self.eval_mapper = eval_mapper
        self.save_path  = Path(save_path)
        self.n_eval     = n_eval
        self.target     = target_reach_rate
        self.patience   = patience
        self.min_evals  = min_evals_before_patience
        self.eval_every = eval_every

        self.best_reach  = -1.0
        self.stagnant    = 0
        self.eval_count  = 0
        self.model_saved = False
        self._last_epoch = -1

    def __call__(self, algo, epoch, total_step):
        if epoch == self._last_epoch:
            return
        self._last_epoch = epoch

        if epoch % self.eval_every != 0:
            return
        self.eval_count += 1

        reach     = _eval_reach_rate(algo, self.eval_env, self.eval_mapper, self.n_eval)
        n_reached = int(reach * self.n_eval)
        improved  = reach > self.best_reach

        if improved:
            self.best_reach = reach
            self.stagnant   = 0
            _save_model_and_scaler(algo, self.save_path)
            self.model_saved = True
            tag = f"NEW BEST -> saved"
        else:
            self.stagnant += 1
            tag = f"no improve {self.stagnant}/{self.patience}"

        print(f"\n[Monitor] epoch={epoch} | step={total_step:,} | "
              f"reach={n_reached}/{self.n_eval} ({reach:.0%}) | {tag}")

        if reach >= self.target:
            print(f"[Monitor] Target {self.target:.0%} reached - stopping early.")
            raise EarlyStopSignal

        if self.eval_count >= self.min_evals and self.stagnant >= self.patience:
            print(f"[Monitor] Plateau ({self.patience} evals no improve) - stopping early.")
            raise EarlyStopSignal

def create_d3rlpy_dataset(data: dict):
    import d3rlpy

    observations = np.clip(data["observations"], -10.0, 10.0)
    actions      = data["actions"]
    rewards      = np.clip(data["rewards"], REWARD_CLIP[0], REWARD_CLIP[1])
    terminals    = data["terminals"]

    return d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

def train_cql(dataset_path: Optional[Path] = None,
              output_dir: Optional[Path] = None,
              n_steps: int = 100_000,
              n_steps_per_epoch: int = 10_000,
              alpha: float = CQL_ALPHA,
              use_gpu: bool = True,
              verbose: bool = True,
              tensorboard_dir: Optional[str] = None,
              save_interval: int = 1) -> Path:
    import d3rlpy
    import torch

    if use_gpu and not torch.cuda.is_available():
        use_gpu = False
        if verbose:
            print("CUDA not available, falling back to CPU")

    gpu_arg = 0 if (use_gpu and torch.cuda.is_available()) else False

    dataset_path = Path(dataset_path or DATA_PROCESSED_DIR / "offline_rl_dataset.npz")
    output_dir   = Path(output_dir or MODELS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading dataset from {dataset_path}")

    data    = DatasetBuilder.load_dataset(dataset_path)
    dataset = create_d3rlpy_dataset(data)

    if verbose:
        print(f"Dataset: {data['observations'].shape[0]} transitions, "
              f"obs_dim={data['observations'].shape[1]}, "
              f"act_dim={data['actions'].shape[1]}, "
              f"episodes={len(dataset.episodes)}")

    cql = d3rlpy.algos.CQL(
        actor_learning_rate=ACTOR_LR,
        critic_learning_rate=CRITIC_LR,
        batch_size=BATCH_SIZE,
        conservative_weight=alpha,
        n_action_samples=10,
        use_gpu=gpu_arg,
        scaler="standard",
        reward_scaler="standard",
    )

    n_epochs = max(1, n_steps // n_steps_per_epoch)
    if verbose:
        gpu_str = "GPU:0" if gpu_arg is not False else "CPU"
        print(f"Training CQL: {n_steps:,} total steps, "
              f"{n_epochs} epochs x {n_steps_per_epoch} steps/epoch "
              f"(alpha={alpha}, device={gpu_str})")

    fit_kwargs = dict(
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        experiment_name="bmds_cql",
        with_timestamp=True,
        save_interval=save_interval,
        verbose=verbose,
        show_progress=verbose,
    )
    if tensorboard_dir is not None:
        fit_kwargs["tensorboard_dir"] = tensorboard_dir

    cql.fit(dataset, **fit_kwargs)

    model_path = output_dir / "bmds_cql_policy.d3"
    cql.save_model(str(model_path))

    if verbose:
        print(f"Model saved to {model_path}")

    return model_path

def train_iql(dataset_path: Optional[Path] = None,
              output_dir: Optional[Path] = None,
              n_steps: int = 100_000,
              n_steps_per_epoch: int = 10_000,
              use_gpu: bool = True,
              verbose: bool = True,
              tensorboard_dir: Optional[str] = None,
              save_interval: int = 1,
              early_stop: bool = True,
              target_reach_rate: float = 0.625) -> Path:
    import d3rlpy
    import torch

    if use_gpu and not torch.cuda.is_available():
        use_gpu = False

    gpu_arg = 0 if (use_gpu and torch.cuda.is_available()) else False

    dataset_path = Path(dataset_path or DATA_PROCESSED_DIR / "offline_rl_dataset.npz")
    output_dir   = Path(output_dir or MODELS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "bmds_iql_policy.d3"

    data    = DatasetBuilder.load_dataset(dataset_path)
    dataset = create_d3rlpy_dataset(data)

    if verbose:
        gpu_str = "GPU:0" if gpu_arg is not False else "CPU"
        print(f"Training IQL: {n_steps:,} total steps, "
              f"{n_steps_per_epoch} per epoch ({gpu_str})")

    iql = d3rlpy.algos.IQL(
        actor_learning_rate=ACTOR_LR,
        critic_learning_rate=CRITIC_LR,
        batch_size=BATCH_SIZE,
        expectile=0.9,
        weight_temp=10.0,
        use_gpu=gpu_arg,
        scaler="standard",
        reward_scaler="standard",
    )

    monitor   = None
    eval_env  = None
    if early_stop:
        from bmds.env.mouse_reach_env import MouseReachEnv
        from bmds.env.sim2screen import Sim2ScreenMapper
        eval_env    = MouseReachEnv()
        eval_mapper = Sim2ScreenMapper()
        monitor = EarlyStopMonitor(
            eval_env, eval_mapper, model_path,
            n_eval=8,
            target_reach_rate=target_reach_rate,
            patience=4,
            min_evals_before_patience=4,
            eval_every=1,
        )
        if verbose:
            print(f"Early-stop monitor: target reach>={target_reach_rate:.0%}, "
                  f"patience=4 epochs, eval every epoch\n")

    fit_kwargs = dict(
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        experiment_name="bmds_iql",
        with_timestamp=True,
        save_interval=save_interval,
        verbose=verbose,
        show_progress=verbose,
        callback=monitor,
    )
    if tensorboard_dir is not None:
        fit_kwargs["tensorboard_dir"] = tensorboard_dir

    try:
        iql.fit(dataset, **fit_kwargs)
    except EarlyStopSignal:
        if verbose:
            print("[EarlyStop] Training halted.")
    finally:
        if eval_env is not None:
            eval_env.close()

    if monitor is None or not monitor.model_saved:
        _save_model_and_scaler(iql, model_path)
        if verbose:
            print(f"IQL model saved to {model_path}")
    else:
        if verbose:
            best_pct = f"{monitor.best_reach:.0%}"
            print(f"IQL best model (reach={best_pct}) saved to {model_path}")

    return model_path

