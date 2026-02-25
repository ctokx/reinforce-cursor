from pathlib import Path
import numpy as np
from bmds.config import DATA_PROCESSED_DIR


def infer_algorithm_from_model_path(model_path: str | Path, default: str = "cql") -> str:
    name = Path(model_path).name.lower()
    if "bmds_bc" in name or "_bc_" in name or name.startswith("bc_"):
        return "bc"
    if "bmds_iql" in name or "_iql_" in name or name.startswith("iql_"):
        return "iql"
    if "bmds_cql" in name or "_cql_" in name or name.startswith("cql_"):
        return "cql"
    return default


def load_policy(model_path: str | Path, algorithm: str = "cql", use_gpu: bool = False):
    import d3rlpy

    model_path = str(model_path)

    if hasattr(d3rlpy, "load_learnable"):
        return d3rlpy.load_learnable(model_path)


    algo_name = algorithm.lower()
    if algo_name == "iql":
        algo = d3rlpy.algos.IQL(use_gpu=use_gpu, scaler="standard", reward_scaler="standard")
    elif algo_name == "bc":
        algo = d3rlpy.algos.BC(use_gpu=use_gpu, scaler="standard")
    else:
        algo = d3rlpy.algos.CQL(use_gpu=use_gpu, scaler="standard", reward_scaler="standard")


    dataset_path = DATA_PROCESSED_DIR / "offline_rl_dataset.npz"
    if dataset_path.exists():
        data = np.load(dataset_path)
        n = min(10_000, data["observations"].shape[0])
        observations = np.clip(data["observations"][:n], -10.0, 10.0).astype(np.float32)
        actions = data["actions"][:n].astype(np.float32)
        rewards = np.clip(data["rewards"][:n], -50.0, 15.0).astype(np.float32)
        terminals = data["terminals"][:n].astype(np.float32)
        if terminals.sum() == 0:
            terminals[-1] = 1.0
    else:
        observations = np.zeros((2, 8), dtype=np.float32)
        actions = np.zeros((2, 2), dtype=np.float32)
        rewards = np.zeros(2, dtype=np.float32)
        terminals = np.array([0.0, 1.0], dtype=np.float32)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations, actions=actions,
        rewards=rewards, terminals=terminals,
    )
    algo.build_with_dataset(dataset)

    if algo.scaler is not None and hasattr(algo.scaler, '_mean') and algo.scaler._mean is None:
        transitions = [t for ep in dataset.episodes for t in ep.transitions]
        if transitions:
            algo.scaler.fit(transitions)

    algo.load_model(model_path)
    return algo
