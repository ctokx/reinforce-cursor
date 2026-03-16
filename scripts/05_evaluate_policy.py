#!/usr/bin/env python3

import sys
import argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from pathlib import Path
from bmds.config import MODELS_DIR
from bmds.data.statistics import load_statistics
from bmds.env.mouse_reach_env import MouseReachEnv
from bmds.env.sim2screen import Sim2ScreenMapper
from bmds.training.evaluate import PolicyEvaluator
from bmds.training.model_loader import (
    load_policy,
    infer_algorithm_from_model_path,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained BMDS policy")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .d3 model file")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--algorithm", choices=["cql", "iql", "bc"], default="cql")
    args = parser.parse_args()

    print("=" * 60)
    print("BMDS Pipeline - Step 5: Evaluate Policy")
    print("=" * 60)

    if args.model:
        model_path = args.model
        algorithm = infer_algorithm_from_model_path(model_path, default=args.algorithm)
    else:
        model_path = str(MODELS_DIR / f"bmds_{args.algorithm}_policy.d3")
        algorithm = args.algorithm
    print(f"Loading model from {model_path}")

    policy = load_policy(model_path, algorithm=algorithm, use_gpu=False)
    stats = load_statistics()
    env = MouseReachEnv()
    mapper = Sim2ScreenMapper()

    evaluator = PolicyEvaluator(env=env, mapper=mapper, human_stats=stats)
    results = evaluator.evaluate(policy, n_episodes=args.episodes)

    env.close()

    print("\nFull results:")
    for key, value in sorted(results.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
