#!/usr/bin/env python
"""Check recorded structure and optional LeRobot readback; not a visual scorer."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py
from burger.data import canonical as C
from burger.data.exporters import writers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--expected_episodes", type=int, required=True)
    parser.add_argument("--lerobot", type=Path)
    args = parser.parse_args()
    report = C.validate_file(args.canonical, require_painter=True)
    if report["num_episodes"] != args.expected_episodes:
        raise ValueError(f"Expected {args.expected_episodes} episodes, found {report['num_episodes']}")
    report["steps_per_episode"] = [ep.num_steps for _, ep in C.iter_episodes(args.canonical)]
    if args.lerobot:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        root = args.lerobot.resolve()
        metadata = json.loads((root / "painter_export.json").read_text())
        if metadata["data_contract"]["convention"] != "painter-absolute-target-v1":
            raise ValueError("Unexpected export convention")
        end_length = metadata["data_contract"]["end_signal"]["final_steps"]
        dataset = LeRobotDataset(repo_id=root.name, root=root, video_backend="pyav")
        if len(dataset) != report["total_steps"]:
            raise ValueError("LeRobot row count differs from canonical data")
        offset = 0
        with h5py.File(args.canonical) as source:
            for name in sorted(source["episodes"], key=lambda value: int(value.split("_")[-1])):
                group = source["episodes"][name]
                length = len(group["actions"])
                state, action = writers.painter_state_action(group, length)
                for local in range(length):
                    row = dataset.hf_dataset[offset + local]
                    end = float(local >= length - end_length)
                    previous = float(local > 0 and local - 1 >= length - end_length)
                    np.testing.assert_allclose(row["observation.state"], np.r_[state[local], previous], atol=1e-6)
                    np.testing.assert_allclose(row["action"], np.r_[action[local], end], atol=1e-6)
                for local in {0, length // 2, length - 1}:
                    frame = dataset[offset + local]
                    image_keys = [key for key in ("observation.image", "observation.wrist_image") if key in frame]
                    if not image_keys:
                        raise ValueError("No decoded camera observations")
                    for key in image_keys:
                        image = np.asarray(frame[key])
                        if image.ndim != 3 or image.shape[0] != 3 or not np.isfinite(image).all():
                            raise ValueError(f"Could not decode {key} as RGB")
                offset += length
        report["lerobot_readback"] = "all numeric rows and each episode's start/middle/end video decoded"
        report["export_metadata"] = metadata
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
