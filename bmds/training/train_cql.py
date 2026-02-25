import numpy as np
from pathlib import Path
from typing import Optional

from bmds.config import (
    CQL_ALPHA, BATCH_SIZE, ACTOR_LR, CRITIC_LR,
    TRAINING_STEPS, MODELS_DIR, DATA_PROCESSED_DIR,
)
from bmds.training.dataset_builder import DatasetBuilder


def create_d3rlpy_dataset(data: dict):
    import d3rlpy

    observations = np.clip(data["observations"], -10.0, 10.0)
    actions = data["actions"]


    rewards = np.clip(data["rewards"], -50.0, 15.0)
    terminals = data["terminals"]

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

    dataset_path = Path(dataset_path or DATA_PROCESSED_DIR / "offline_rl_dataset.npz")
    output_dir = Path(output_dir or MODELS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading dataset from {dataset_path}")

    data = DatasetBuilder.load_dataset(dataset_path)
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
        use_gpu=use_gpu,
        scaler="standard",
        reward_scaler="standard",
    )

    n_epochs = max(1, n_steps // n_steps_per_epoch)
    if verbose:
        gpu_str = "GPU" if use_gpu else "CPU"
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
              save_interval: int = 1) -> Path:
    import d3rlpy
    import torch

    if use_gpu and not torch.cuda.is_available():
        use_gpu = False

    dataset_path = Path(dataset_path or DATA_PROCESSED_DIR / "offline_rl_dataset.npz")
    output_dir = Path(output_dir or MODELS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = DatasetBuilder.load_dataset(dataset_path)
    dataset = create_d3rlpy_dataset(data)

    if verbose:
        print(f"Training IQL: {n_steps:,} total steps, {n_steps_per_epoch} per epoch")

    iql = d3rlpy.algos.IQL(
        actor_learning_rate=ACTOR_LR,
        critic_learning_rate=CRITIC_LR,
        batch_size=BATCH_SIZE,
        expectile=0.7,
        weight_temp=3.0,
        use_gpu=use_gpu,
        scaler="standard",
        reward_scaler="standard",
    )

    fit_kwargs = dict(
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        experiment_name="bmds_iql",
        with_timestamp=True,
        save_interval=save_interval,
        verbose=verbose,
        show_progress=verbose,
    )
    if tensorboard_dir is not None:
        fit_kwargs["tensorboard_dir"] = tensorboard_dir

    iql.fit(dataset, **fit_kwargs)

    model_path = output_dir / "bmds_iql_policy.d3"
    iql.save_model(str(model_path))

    if verbose:
        print(f"IQL model saved to {model_path}")

    return model_path


def train_bc(dataset_path: Optional[Path] = None,
             output_dir: Optional[Path] = None,
             n_steps: int = 50_000,
             n_steps_per_epoch: int = 5_000,
             use_gpu: bool = True,
             verbose: bool = True,
             tensorboard_dir: Optional[str] = None,
             save_interval: int = 1) -> Path:
    import d3rlpy
    import torch

    if use_gpu and not torch.cuda.is_available():
        use_gpu = False

    dataset_path = Path(dataset_path or DATA_PROCESSED_DIR / "offline_rl_dataset.npz")
    output_dir = Path(output_dir or MODELS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = DatasetBuilder.load_dataset(dataset_path)
    dataset = create_d3rlpy_dataset(data)

    if verbose:
        gpu_str = "GPU" if use_gpu else "CPU"
        print(f"Training BC: {n_steps:,} total steps ({gpu_str})")

    bc = d3rlpy.algos.BC(
        learning_rate=ACTOR_LR,
        batch_size=BATCH_SIZE,
        policy_type="deterministic",
        use_gpu=use_gpu,
        scaler="standard",
    )

    fit_kwargs = dict(
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        experiment_name="bmds_bc",
        with_timestamp=True,
        save_interval=save_interval,
        verbose=verbose,
        show_progress=verbose,
    )
    if tensorboard_dir is not None:
        fit_kwargs["tensorboard_dir"] = tensorboard_dir

    bc.fit(dataset, **fit_kwargs)

    model_path = output_dir / "bmds_bc_policy.d3"
    bc.save_model(str(model_path))

    if verbose:
        print(f"BC model saved to {model_path}")

    return model_path
