#!/usr/bin/env python
"""Convert a LeRobotDataset to a RobotWin/DP3-style Zarr dataset.

This script reads frames from `lerobot.datasets.lerobot_dataset.LeRobotDataset` and writes a
Zarr directory with this structure:

- <output>.zarr/
  - data/
    - point_cloud     (T, ..., 6)  float32 (if present)
    - state           (T, D)       float32 (if present)
    - action          (T, A)       float32 (if present)
    - eef             (T, E)       float32 (optional, if present)
    - reward          (T,)         float32 (optional, if present)
    - done            (T,)         int8    (optional, if present)
    - timestamp       (T,)         float64/int64 (optional, if present)
    - frame_index     (T,)         int64   (optional, if present)
    - episode_index   (T,)         int64   (optional, if present)
  - meta/
    - episode_ends    (num_episodes,) int64

It matches the append/resize pattern used in `res/robotwin_DP3/scripts/process_data_safe.py`.

Example:
  python third_party/lerobot/src/lerobot/scripts/lerobot_to_zarr.py \
    --repo-id lerobot/pusht \
    --episodes 0,1,2 \
    --output ./data/pusht_0_1_2.zarr
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import tqdm
import zarr

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_EEF, OBS_POINTCLOUD, OBS_STATE, REWARD


@dataclass(frozen=True)
class ZarrSpec:
    key: str
    dtype: str
    shape_tail: tuple[int, ...]


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


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.float32:
        return x
    return x.astype(np.float32, copy=False)


def _tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(type(x))


def _infer_specs(sample: dict) -> dict[str, ZarrSpec]:
    specs: dict[str, ZarrSpec] = {}

    def add_if_present(src_key: str, dst_key: str, dtype: str) -> None:
        if src_key not in sample:
            return
        val = sample[src_key]
        if not hasattr(val, "shape"):
            return
        shape_tail = tuple(int(d) for d in val.shape)
        specs[dst_key] = ZarrSpec(key=src_key, dtype=dtype, shape_tail=shape_tail)

    add_if_present(OBS_POINTCLOUD, "point_cloud", "float32")
    add_if_present(OBS_STATE, "state", "float32")
    add_if_present(ACTION, "action", "float32")

    # optional extras
    add_if_present(OBS_EEF, "eef", "float32")
    add_if_present(REWARD, "reward", "float32")
    add_if_present(DONE, "done", "int8")

    add_if_present("timestamp", "timestamp", "float64")
    add_if_present("frame_index", "frame_index", "int64")
    add_if_present("episode_index", "episode_index", "int64")

    return specs


def _create_zarr_arrays(
    zarr_data: zarr.hierarchy.Group,
    specs: dict[str, ZarrSpec],
    compressor: zarr.Codec,
    chunk_len: int,
) -> dict[str, zarr.Array]:
    arrays: dict[str, zarr.Array] = {}
    for dst_key, spec in specs.items():
        arrays[dst_key] = zarr_data.create_dataset(
            dst_key,
            shape=(0,) + spec.shape_tail,
            chunks=(chunk_len,) + spec.shape_tail,
            dtype=spec.dtype,
            compressor=compressor,
        )
    return arrays


def _append(arr: zarr.Array, batch_np: np.ndarray) -> None:
    old_size = arr.shape[0]
    new_size = old_size + batch_np.shape[0]
    arr.resize(new_size, *arr.shape[1:])
    arr[old_size:new_size] = batch_np


def convert_lerobot_to_zarr(
    dataset: LeRobotDataset,
    output: Path,
    batch_size: int,
    num_workers: int,
    chunk_len: int,
    overwrite: bool,
    include_optional: bool,
) -> Path:
    output = Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)

    # Prepare zarr structure
    zroot = zarr.group(str(output))
    zdata = zroot.create_group("data")
    zmeta = zroot.create_group("meta")

    # Infer shapes/dtypes
    sample = dataset[0]
    specs = _infer_specs(sample)

    required = {"point_cloud", "state", "action"}
    missing = sorted(required - set(specs.keys()))
    if missing:
        raise ValueError(
            "Dataset is missing required keys for DP3-style export: "
            + ", ".join(missing)
            + f". Available keys: {sorted(sample.keys())}"
        )

    if not include_optional:
        for k in ["eef", "reward", "done", "timestamp", "frame_index", "episode_index"]:
            specs.pop(k, None)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    arrays = _create_zarr_arrays(zdata, specs, compressor=compressor, chunk_len=chunk_len)

    # Store some metadata
    zmeta.attrs["repo_id"] = dataset.repo_id
    zmeta.attrs["fps"] = int(dataset.fps)
    if getattr(dataset.meta, "robot_type", None) is not None:
        zmeta.attrs["robot_type"] = dataset.meta.robot_type
    if getattr(dataset, "episodes", None) is not None:
        # may exist depending on LeRobotDataset implementation
        try:
            zmeta.attrs["requested_episodes"] = list(dataset.episodes)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Iterate and write
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    episode_ends: list[int] = []
    total_count = 0
    prev_episode_index: int | None = None

    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="lerobot->zarr"):
        # DataLoader default collate returns torch tensors.
        bsz = int(batch["index"].shape[0])

        # Track episode boundaries at per-frame granularity.
        if "episode_index" in batch:
            ep_idx_np = _tensor_to_numpy(batch["episode_index"]).reshape(-1)
            for i in range(bsz):
                ep_idx = int(ep_idx_np[i])
                if prev_episode_index is None:
                    prev_episode_index = ep_idx
                elif ep_idx != prev_episode_index:
                    episode_ends.append(total_count)
                    prev_episode_index = ep_idx
                total_count += 1
        else:
            # Fallback: assume single-episode dataset.
            total_count += bsz

        # Write each exported key.
        for dst_key, spec in specs.items():
            x = batch[spec.key]
            x_np = _tensor_to_numpy(x)

            # Scalar tensors sometimes come as shape (B,) or (B,1); keep whatever tail we inferred.
            # Ensure batch dimension exists.
            if x_np.ndim == len(spec.shape_tail):
                x_np = x_np.reshape((bsz,) + spec.shape_tail)

            if spec.dtype == "float32":
                x_np = _ensure_float32(x_np)
            elif spec.dtype == "int64":
                x_np = x_np.astype(np.int64, copy=False)
            elif spec.dtype == "int8":
                x_np = x_np.astype(np.int8, copy=False)

            _append(arrays[dst_key], x_np)

    # Close last episode
    if prev_episode_index is not None:
        episode_ends.append(total_count)

    zmeta.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LeRobotDataset to zarr (RobotWin/DP3-style).")

    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Root directory for the dataset stored locally. If omitted, uses HuggingFace cache and may download."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help=(
            "Episode indices to convert, e.g. '0,1,2' or '0-9'. If omitted, converts all episodes."
        ),
    )

    parser.add_argument("--output", type=Path, required=True, help="Output zarr directory, e.g. ./data/out.zarr")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Passed to LeRobotDataset(tolerance_s=...).",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default=None,
        help="Passed to LeRobotDataset(video_backend=...).",
    )

    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also export optional keys if present (eef/reward/done/timestamp/frame_index/episode_index).",
    )

    args = parser.parse_args()

    episodes = _parse_episodes_arg(args.episodes)
    dataset = LeRobotDataset(
        args.repo_id,
        episodes=episodes,
        root=args.root,
        tolerance_s=args.tolerance_s,
        video_backend=args.video_backend,
    )

    convert_lerobot_to_zarr(
        dataset=dataset,
        output=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunk_len=args.chunk_len,
        overwrite=args.overwrite,
        include_optional=args.include_optional,
    )


if __name__ == "__main__":
    main()
