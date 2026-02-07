#!/usr/bin/env python
"""Convert AutoMoMa HDF5 episodes to LeRobot format.

Reads AutoMoMa-style HDF5 files with structure documented in hdf5_stucture.md
and writes a LeRobotDataset on disk.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass(frozen=True)
class EpisodeData:
    state: np.ndarray
    action: np.ndarray
    eef: np.ndarray | None
    rgb: dict[str, np.ndarray]
    depth: dict[str, np.ndarray]
    pointcloud: np.ndarray | None
    robot_name: str | None


def _parse_episodes_arg(episodes: str | None) -> list[int] | None:
    if episodes is None:
        return None
    episodes = episodes.strip()
    if not episodes:
        return None
    out: list[int] = []
    for part in episodes.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a)
            end = int(b)
            if end < start:
                raise ValueError(f"Invalid episode range: {part}")
            out.extend(list(range(start, end + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _load_episode(
    path: Path,
    mobile_base_mode: str,
    joint_order: list[str],
) -> EpisodeData:
    with h5py.File(path, "r") as root:
        robot_name = None
        if "env_info" in root and "robot_name" in root["env_info"]:
            robot_name = root["env_info"]["robot_name"][()].decode("utf-8")

        obs = root["obs"]
        joint = obs["joint"]

        joint_states: dict[str, np.ndarray] = {}
        if "arm" in joint:
            joint_states["arm"] = _ensure_2d(np.asarray(joint["arm"]))
        if "gripper" in joint:
            joint_states["gripper"] = _ensure_2d(np.asarray(joint["gripper"]))
        if "mobile_base" in joint:
            joint_states["mobile_base"] = _ensure_2d(np.asarray(joint["mobile_base"]))

        if not joint_states:
            raise ValueError(f"No joint data found in {path}")

        # Build state in requested order.
        state_parts = []
        for name in joint_order:
            if name not in joint_states:
                continue
            arr = joint_states[name]
            if name == "arm":
                arr = arr[:, :7]
            elif name == "mobile_base":
                arr = arr[:, :3]
            state_parts.append(arr)

        if not state_parts:
            raise ValueError(f"No joint data matched order in {path}")

        state = np.concatenate(state_parts, axis=1)

        # Compute actions to match DP3 convention.
        if mobile_base_mode == "absolute":
            action = state[1:]
        elif mobile_base_mode == "relative":
            action_parts = []
            for name in joint_order:
                if name not in joint_states:
                    continue
                arr = joint_states[name]
                if name == "arm":
                    arr = arr[:, :7]
                elif name == "mobile_base":
                    arr = arr[:, :3]
                if name == "mobile_base":
                    action_parts.append(arr[1:] - arr[:-1])
                else:
                    action_parts.append(arr[1:])
            action = np.concatenate(action_parts, axis=1)
        else:
            raise ValueError("mobile_base_mode must be 'relative' or 'absolute'")

        # Align all modalities to action length.
        valid_len = action.shape[0]
        state = state[:valid_len]

        eef = None
        if "eef" in obs:
            eef = np.asarray(obs["eef"])[:valid_len]

        rgb: dict[str, np.ndarray] = {}
        depth: dict[str, np.ndarray] = {}

        if "rgb" in obs:
            for cam_name in obs["rgb"].keys():
                rgb[cam_name] = np.asarray(obs["rgb"][cam_name])[:valid_len]

        if "depth" in obs:
            for cam_name in obs["depth"].keys():
                depth[cam_name] = np.asarray(obs["depth"][cam_name])[:valid_len]

        pointcloud = None
        if "point_cloud" in obs:
            pointcloud = np.asarray(obs["point_cloud"])[:valid_len]

        return EpisodeData(
            state=state,
            action=action,
            eef=eef,
            rgb=rgb,
            depth=depth,
            pointcloud=pointcloud,
            robot_name=robot_name,
        )


def _infer_features(sample: EpisodeData, use_videos: bool) -> dict:
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (sample.state.shape[1],),
            "names": [[f"state_{i}" for i in range(sample.state.shape[1])]],
        },
        "action": {
            "dtype": "float32",
            "shape": (sample.action.shape[1],),
            "names": [[f"action_{i}" for i in range(sample.action.shape[1])]],
        },
    }

    if sample.eef is not None:
        features["observation.eef"] = {
            "dtype": "float32",
            "shape": (sample.eef.shape[1],),
            "names": [[f"eef_{i}" for i in range(sample.eef.shape[1])]],
        }

    for cam_name, frames in sample.rgb.items():
        h, w = frames.shape[1], frames.shape[2]
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        }

    for cam_name, frames in sample.depth.items():
        h, w = frames.shape[1], frames.shape[2]
        features[f"observation.depth.{cam_name}"] = {
            "dtype": "float32",
            "shape": (1, h, w),
            "names": ["channels", "height", "width"],
        }

    if sample.pointcloud is not None:
        features["observation.pointcloud"] = {
            "dtype": "float32",
            "shape": sample.pointcloud.shape[1:],
            "names": ["points", "xyzrgb"],
        }

    return features


def _iter_episode_paths(input_dir: Path, episodes: list[int] | None) -> list[Path]:
    if episodes is None:
        return sorted(input_dir.glob("episode*.hdf5"))
    return [input_dir / f"episode{ep:06d}.hdf5" for ep in episodes]


def _add_episode_frames(dataset: LeRobotDataset, ep: EpisodeData, task: str) -> None:
    length = ep.action.shape[0]
    for t in range(length):
        frame = {
            "observation.state": ep.state[t].astype(np.float32, copy=False),
            "action": ep.action[t].astype(np.float32, copy=False),
        }

        if ep.eef is not None:
            frame["observation.eef"] = ep.eef[t].astype(np.float32, copy=False)

        for cam_name, frames in ep.rgb.items():
            img = frames[t]
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.transpose(2, 0, 1)
            frame[f"observation.images.{cam_name}"] = img

        for cam_name, frames in ep.depth.items():
            d = frames[t]
            if d.ndim == 2:
                d = d[np.newaxis, ...]
            frame[f"observation.depth.{cam_name}"] = d.astype(np.float32, copy=False)

        if ep.pointcloud is not None:
            frame["observation.pointcloud"] = ep.pointcloud[t].astype(np.float32, copy=False)

        frame["task"] = task

        dataset.add_frame(frame)

    dataset.save_episode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AutoMoMa HDF5 dataset to LeRobot format.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with episode*.hdf5")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot repo id, e.g. automoma/my_task")
    parser.add_argument("--root", type=Path, default=None, help="Output root directory for LeRobot dataset")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--episodes", type=str, default=None, help="Episode indices: '0,1,2' or '0-9'")
    parser.add_argument(
        "--mobile-base-mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
    )
    parser.add_argument(
        "--joint-order",
        type=str,
        default="arm,gripper,mobile_base",
        help="Comma-separated joint order for state/action",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="default",
        help="Task label stored per frame (required by LeRobot validation)",
    )

    args = parser.parse_args()

    episodes = _parse_episodes_arg(args.episodes)
    joint_order = [name.strip() for name in args.joint_order.split(",") if name.strip()]

    episode_paths = _iter_episode_paths(args.input_dir, episodes)
    if not episode_paths:
        raise SystemExit(f"No episodes found under {args.input_dir}")

    first_ep = _load_episode(episode_paths[0], args.mobile_base_mode, joint_order)
    features = _infer_features(first_ep, use_videos=args.use_videos)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        features=features,
        robot_type=first_ep.robot_name,
        use_videos=args.use_videos,
    )

    for path in tqdm.tqdm(episode_paths, desc="hdf5->lerobot"):
        ep = _load_episode(path, args.mobile_base_mode, joint_order)
        _add_episode_frames(dataset, ep, task=args.task)

    dataset.finalize()


if __name__ == "__main__":
    main()
