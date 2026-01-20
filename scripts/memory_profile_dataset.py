#!/usr/bin/env python3
"""Profile process RSS when preloading a LeRobot dataset.

Usage:
  python scripts/memory_profile_dataset.py /path/to/dataset --keys pointcloud

Keys:
  pointcloud  - only `observation.pointcloud`
  images      - only image/video observation keys present in info.json
  all         - load all keys (default)
  cam=<name>   - load a specific camera key, e.g. cam=observation.images.ego_topdown
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import psutil
import logging

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _read_info(root: Path) -> dict | None:
    for p in (root / "meta" / "info.json", root / "info.json"):
        if p.exists():
            return json.loads(p.read_text())
    return None


def _sizeof_value(value) -> int:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.element_size() * value.nelement()
    except Exception:
        pass

    if isinstance(value, np.ndarray):
        return value.nbytes

    if isinstance(value, (bytes, bytearray)):
        return len(value)

    if isinstance(value, str):
        return len(value.encode("utf-8"))

    if isinstance(value, dict):
        return sum(_sizeof_value(v) for v in value.values())

    if isinstance(value, (list, tuple)):
        return sum(_sizeof_value(v) for v in value)

    return 0


def _estimate_item_sizes(ds: LeRobotDataset, sample_size: int) -> dict:
    total_items = len(ds)
    if total_items == 0:
        return {
            "sample_size": 0,
            "avg_item_mb": 0.0,
            "estimated_total_gb": 0.0,
            "per_key_avg_mb": {},
        }

    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    if sample_size >= total_items:
        indices = list(range(total_items))
    else:
        indices = np.linspace(0, total_items - 1, num=sample_size, dtype=int).tolist()
        indices = sorted(set(indices))

    per_key_bytes: dict[str, int] = {}
    total_bytes = 0
    count = 0

    for idx in indices:
        item = ds[idx]
        item_bytes = 0
        for key, value in item.items():
            value_bytes = _sizeof_value(value)
            item_bytes += value_bytes
            per_key_bytes[key] = per_key_bytes.get(key, 0) + value_bytes
        total_bytes += item_bytes
        count += 1

    per_key_avg_mb = {k: round(v / count / (1024**2), 4) for k, v in per_key_bytes.items()}
    avg_item_mb = (total_bytes / count) / (1024**2)
    estimated_total_gb = (avg_item_mb * total_items) / 1024

    return {
        "sample_size": count,
        "avg_item_mb": round(avg_item_mb, 4),
        "estimated_total_gb": round(estimated_total_gb, 4),
        "per_key_avg_mb": dict(sorted(per_key_avg_mb.items(), key=lambda x: -x[1])),
    }


def profile_dataset(
    root: Path,
    keys_arg: str | None,
    sample_size: int,
    estimate_only: bool,
    estimate: bool,
):
    root = Path(root)
    info = _read_info(root)
    # choose repo_id fallback
    repo_id = info.get("repo_id") if info and "repo_id" in info else root.name

    # determine requested_keys set
    requested_keys = None
    if keys_arg is None or keys_arg == "all":
        requested_keys = None
    elif keys_arg == "pointcloud":
        requested_keys = {"observation.pointcloud"}
    elif keys_arg == "images":
        if info is None:
            raise SystemExit("info.json not found in dataset root; cannot discover image keys")
        requested_keys = {k for k, v in info.get("features", {}).items() if v.get("dtype") in ("image", "video")}
    elif keys_arg.startswith("cam="):
        requested_keys = {keys_arg.split("=", 1)[1]}
    else:
        raise SystemExit(f"Unknown keys argument: {keys_arg}")

    report = {
        "root": str(root),
        "repo_id": repo_id,
        "requested_keys": sorted(requested_keys) if requested_keys is not None else None,
    }

    if estimate_only or estimate:
        ds = LeRobotDataset(repo_id, root=root, preload=False, requested_keys=requested_keys, download_videos=False)
        report["estimate"] = _estimate_item_sizes(ds, sample_size)

        if estimate_only:
            print(json.dumps(report, indent=2))
            return

    proc = psutil.Process()
    before = proc.memory_info().rss / (1024**3)
    t0 = time.time()

    ds = LeRobotDataset(repo_id, root=root, preload=True, requested_keys=requested_keys, download_videos=False)

    t1 = time.time()
    after = proc.memory_info().rss / (1024**3)

    report.update(
        {
            "total_episodes": ds.meta.total_episodes,
            "total_frames": ds.meta.total_frames,
            "preload_time_s": round(t1 - t0, 3),
            "rss_gb_before": round(before, 4),
            "rss_gb_after": round(after, 4),
            "rss_gb_delta": round(after - before, 4),
            "num_preloaded_items": len(ds._preloaded_items) if ds._preloaded_items is not None else 0,
            "per_item_delta_mb": None,
        }
    )

    if report["num_preloaded_items"] > 0:
        report["per_item_delta_mb"] = round(
            (report["rss_gb_delta"] * 1024) / report["num_preloaded_items"], 3
        )

    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Path to local dataset root (contains meta/info.json)")
    parser.add_argument("--keys", default="all", help="Which keys to preload: pointcloud|images|all|cam=<key>")
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Number of items to sample for size estimation (used with --estimate-only or --estimate)",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Skip full preload and only estimate per-key/item memory from samples",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Also compute per-key/item memory estimate before preloading",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    profile_dataset(Path(args.dataset_root), args.keys, args.sample, args.estimate_only, args.estimate)


if __name__ == "__main__":
    main()
