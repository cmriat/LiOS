"""Convert LeRobot stats.json to Pi norm_stats.json format."""

import json
import pathlib

import numpy as np

from pi.shared import normalize


def convert_lerobot_stats_to_norm_stats(lerobot_stats_path: str, output_dir: str) -> None:
    """将LeRobot格式的stats.json转换为Pi模型期望的norm_stats.json格式.

    LeRobot格式:
    {
        "observation.state": {"mean": [...], "std": [...], "min": [...], "max": [...], "q01": [...], "q99": [...]},
        "action": {"mean": [...], "std": [...], ...}
    }

    Pi格式:
    {
        "norm_stats": {
            "state": {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
            "actions": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}
        }
    }
    """
    with open(lerobot_stats_path, "r") as f:
        lerobot_stats = json.load(f)

    norm_stats = {}

    # Key mapping: LeRobot format → Pi format
    key_mapping = {"observation.state": "state", "action": "actions"}

    for lerobot_key, pi_key in key_mapping.items():
        if lerobot_key not in lerobot_stats:
            print(f"Warning: {lerobot_key} not found in stats.json, skipping.")
            continue

        stats = lerobot_stats[lerobot_key]

        norm_stats[pi_key] = normalize.NormStats(
            mean=np.array(stats["mean"], dtype=np.float32),
            std=np.array(stats["std"], dtype=np.float32),
            q01=np.array(stats["q01"], dtype=np.float32) if "q01" in stats else None,
            q99=np.array(stats["q99"], dtype=np.float32) if "q99" in stats else None,
        )

        print(f"Converted {lerobot_key} -> {pi_key}:")
        print(f"  mean shape: {norm_stats[pi_key].mean.shape}")
        print(f"  std shape: {norm_stats[pi_key].std.shape}")
        if norm_stats[pi_key].q01 is not None:
            print(f"  q01 shape: {norm_stats[pi_key].q01.shape}")
        if norm_stats[pi_key].q99 is not None:
            print(f"  q99 shape: {norm_stats[pi_key].q99.shape}")

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalize.save(output_path, norm_stats)

    print(f"\n✓ Successfully converted and saved norm_stats.json to {output_path}")
    print(f"  Output file: {output_path / 'norm_stats.json'}")


if __name__ == "__main__":
    lerobot_stats_path = "/path/to/lerobot_dataset/meta/stats.json"
    output_dir = "./assets/pi05_airbot/airbot"

    convert_lerobot_stats_to_norm_stats(lerobot_stats_path, output_dir)
