#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skeletonize Pointcept-exported point clouds with pc-skeletor.

This script assumes Pointcept part-segmentation PLY exports encode semantic
labels as RGB colors. In the default fallback mode, each unique RGB triplet is
treated as one pseudo-instance and skeletonized independently with LBC.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RgbTuple = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Skeletonize a Pointcept-exported prediction PLY. By default each "
            "unique RGB color is treated as one pseudo-instance."
        )
    )
    parser.add_argument("--input-ply", type=Path, required=True, help="Input Pointcept PLY file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for skeleton outputs.")
    parser.add_argument(
        "--instance-mode",
        type=str,
        choices=("color_label", "external_instance"),
        default="color_label",
        help="How to split the input point cloud into instances.",
    )
    parser.add_argument(
        "--instance-file",
        type=Path,
        default=None,
        help=(
            "Per-point instance ids for external_instance mode. Supported formats: "
            ".npy, .json, .txt, .csv."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("lbc", "slbc"),
        default="lbc",
        help="Whole-point-cloud skeletonization method. Instance skeletons always use LBC.",
    )
    parser.add_argument(
        "--down-sample",
        type=float,
        default=0.01,
        help="Voxel size passed to pc-skeletor down_sample.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=32,
        help="Skip instances with fewer than this many points.",
    )
    parser.add_argument(
        "--trunk-colors",
        nargs="*",
        default=None,
        help=(
            "Optional RGB triplets such as 230,25,75 60,180,75 for whole-cloud "
            "SLBC. If omitted, the largest color group is used as trunk."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable more detailed logging from this wrapper and pc-skeletor.",
    )
    return parser.parse_args()


def parse_rgb_triplet(value: str) -> RgbTuple:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"Expected RGB triplet r,g,b but got: {value}")
    rgb = tuple(int(part) for part in parts)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"RGB values must be in [0, 255], got: {value}")
    return rgb  # type: ignore[return-value]


def load_ascii_ply_with_rgb(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertex_count = None
    vertex_properties: List[str] = []
    in_vertex_element = False
    data_start_idx = None

    lines = file_path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if idx == 0 and stripped != "ply":
            raise ValueError(f"Not a PLY file: {file_path}")
        if stripped.startswith("format "):
            if "ascii" not in stripped:
                raise ValueError(
                    f"Only ASCII PLY is supported for input parsing, got header: {stripped}"
                )
        elif stripped.startswith("element "):
            parts = stripped.split()
            if len(parts) >= 3 and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex_element = True
                vertex_properties = []
            else:
                in_vertex_element = False
        elif stripped.startswith("property ") and in_vertex_element:
            property_name = stripped.split()[-1]
            vertex_properties.append(property_name)
        elif stripped == "end_header":
            data_start_idx = idx + 1
            break

    if vertex_count is None or data_start_idx is None:
        raise ValueError(f"PLY header is missing vertex metadata: {file_path}")

    required = ("x", "y", "z", "red", "green", "blue")
    missing = [name for name in required if name not in vertex_properties]
    if missing:
        raise ValueError(f"PLY is missing required properties {missing}: {file_path}")

    prop_to_idx = {name: idx for idx, name in enumerate(vertex_properties)}
    xyz = np.empty((vertex_count, 3), dtype=np.float32)
    rgb = np.empty((vertex_count, 3), dtype=np.uint8)

    if len(lines) < data_start_idx + vertex_count:
        raise ValueError(
            f"PLY vertex section is shorter than declared vertex count ({vertex_count}): {file_path}"
        )

    for point_idx in range(vertex_count):
        parts = lines[data_start_idx + point_idx].split()
        if len(parts) < len(vertex_properties):
            raise ValueError(
                f"Vertex line {point_idx} has {len(parts)} values but expected "
                f"{len(vertex_properties)} in {file_path}"
            )
        xyz[point_idx, 0] = float(parts[prop_to_idx["x"]])
        xyz[point_idx, 1] = float(parts[prop_to_idx["y"]])
        xyz[point_idx, 2] = float(parts[prop_to_idx["z"]])
        rgb[point_idx, 0] = int(parts[prop_to_idx["red"]])
        rgb[point_idx, 1] = int(parts[prop_to_idx["green"]])
        rgb[point_idx, 2] = int(parts[prop_to_idx["blue"]])

    return xyz, rgb


def load_external_instance_ids(file_path: Path, expected_points: int) -> np.ndarray:
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        instance_ids = np.load(file_path)
    elif suffix in {".txt", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        instance_ids = np.loadtxt(file_path, delimiter=delimiter)
    elif suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "instance_ids" not in payload:
                raise ValueError(
                    f"JSON instance file must contain an 'instance_ids' field: {file_path}"
                )
            payload = payload["instance_ids"]
        instance_ids = np.asarray(payload)
    else:
        raise ValueError(
            f"Unsupported instance file format '{suffix}'. Use .npy, .json, .txt, or .csv."
        )

    instance_ids = np.asarray(instance_ids).reshape(-1)
    if instance_ids.shape[0] != expected_points:
        raise ValueError(
            f"Instance ids length mismatch: expected {expected_points}, got {instance_ids.shape[0]}"
        )
    return instance_ids.astype(np.int64)


def make_point_cloud(points: np.ndarray, colors: np.ndarray) -> "o3d.geometry.PointCloud":
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    return pcd


def group_by_color(colors: np.ndarray) -> List[Dict[str, object]]:
    unique_colors, inverse, counts = np.unique(colors, axis=0, return_inverse=True, return_counts=True)
    groups: List[Dict[str, object]] = []
    order = np.lexsort((unique_colors[:, 2], unique_colors[:, 1], unique_colors[:, 0]))
    for color_idx in order.tolist():
        color = tuple(int(value) for value in unique_colors[color_idx].tolist())
        point_indices = np.flatnonzero(inverse == color_idx)
        groups.append(
            {
                "name": f"color_{color[0]}_{color[1]}_{color[2]}",
                "color": color,
                "indices": point_indices,
                "num_points": int(counts[color_idx]),
            }
        )
    return groups


def group_by_external_instances(instance_ids: np.ndarray) -> List[Dict[str, object]]:
    unique_ids, counts = np.unique(instance_ids, return_counts=True)
    groups: List[Dict[str, object]] = []
    for instance_id, count in sorted(zip(unique_ids.tolist(), counts.tolist()), key=lambda item: item[0]):
        point_indices = np.flatnonzero(instance_ids == instance_id)
        groups.append(
            {
                "name": f"instance_{int(instance_id)}",
                "instance_id": int(instance_id),
                "indices": point_indices,
                "num_points": int(count),
            }
        )
    return groups


def export_result_bundle(
    output_dir: Path,
    input_pcd: "o3d.geometry.PointCloud",
    skeletonizer,
    metadata: Dict[str, object],
) -> None:
    import networkx as nx
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_dir / "input_points.ply"), input_pcd)
    o3d.io.write_point_cloud(str(output_dir / "skeleton.ply"), skeletonizer.skeleton)
    o3d.io.write_line_set(str(output_dir / "topology.ply"), skeletonizer.topology)
    nx.write_gpickle(skeletonizer.skeleton_graph, str(output_dir / "skeleton_graph.gpickle"))
    nx.write_gpickle(skeletonizer.topology_graph, str(output_dir / "topology_graph.gpickle"))
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_lbc(point_cloud: "o3d.geometry.PointCloud", down_sample: float, verbose: bool = False):
    from pc_skeletor import LBC

    skeletonizer = LBC(
        point_cloud=point_cloud,
        down_sample=down_sample,
        filter_nb_neighbors=False,
        filter_std_ratio=False,
        debug=False,
        verbose=verbose,
    )
    skeletonizer.extract_skeleton()
    skeletonizer.extract_topology()
    return skeletonizer


def infer_trunk_colors(color_groups: Sequence[Dict[str, object]]) -> List[RgbTuple]:
    largest_group = max(color_groups, key=lambda group: int(group["num_points"]))
    return [largest_group["color"]]  # type: ignore[list-item]


def run_slbc(
    points: np.ndarray,
    colors: np.ndarray,
    trunk_colors: Sequence[RgbTuple],
    down_sample: float,
    verbose: bool = False,
):
    from pc_skeletor import SLBC

    trunk_colors_array = np.asarray(trunk_colors, dtype=np.uint8)
    matches = np.all(colors[:, None, :] == trunk_colors_array[None, :, :], axis=2)
    trunk_mask = np.any(matches, axis=1)

    if not np.any(trunk_mask):
        raise ValueError("SLBC trunk selection is empty.")
    if np.all(trunk_mask):
        raise ValueError("SLBC trunk selection covers the whole point cloud; branches are empty.")

    trunk_pcd = make_point_cloud(points[trunk_mask], colors[trunk_mask])
    branch_pcd = make_point_cloud(points[~trunk_mask], colors[~trunk_mask])

    skeletonizer = SLBC(
        point_cloud={"trunk": trunk_pcd, "branches": branch_pcd},
        down_sample=down_sample,
        filter_nb_neighbors=False,
        filter_std_ratio=False,
        debug=False,
        verbose=verbose,
    )
    skeletonizer.extract_skeleton()
    skeletonizer.extract_topology()
    return skeletonizer


def summarize_group(group: Dict[str, object]) -> Dict[str, object]:
    summary = {
        "name": group["name"],
        "num_points": int(group["num_points"]),
    }
    if "color" in group:
        summary["color"] = list(group["color"])  # type: ignore[arg-type]
    if "instance_id" in group:
        summary["instance_id"] = int(group["instance_id"])
    return summary


def skeletonize_instances(
    points: np.ndarray,
    colors: np.ndarray,
    groups: Sequence[Dict[str, object]],
    output_root: Path,
    input_ply: Path,
    down_sample: float,
    min_points: int,
    verbose: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    instances_root = output_root / "instances"
    instances_root.mkdir(parents=True, exist_ok=True)

    for group in groups:
        result = summarize_group(group)
        indices = group["indices"]  # type: ignore[assignment]
        instance_points = points[indices]
        instance_colors = colors[indices]
        result["output_dir"] = str((instances_root / str(group["name"])).resolve())
        result["method"] = "lbc"

        if int(group["num_points"]) < min_points:
            result["status"] = "skipped"
            result["reason"] = f"num_points < min_points ({group['num_points']} < {min_points})"
            results.append(result)
            continue

        try:
            instance_pcd = make_point_cloud(instance_points, instance_colors)
            skeletonizer = run_lbc(instance_pcd, down_sample=down_sample, verbose=verbose)
            output_dir = instances_root / str(group["name"])
            export_result_bundle(
                output_dir=output_dir,
                input_pcd=instance_pcd,
                skeletonizer=skeletonizer,
                metadata={
                    **result,
                    "status": "ok",
                    "input_file": str(input_ply.resolve()),
                    "down_sample": down_sample,
                    "min_points": min_points,
                },
            )
            result["status"] = "ok"
        except Exception as exc:  # pragma: no cover - integration behavior
            result["status"] = "failed"
            result["reason"] = str(exc)
            logging.exception("Instance skeletonization failed for %s", group["name"])

        results.append(result)

    return results


def skeletonize_full_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    color_groups: Sequence[Dict[str, object]],
    output_root: Path,
    input_ply: Path,
    method: str,
    down_sample: float,
    trunk_colors: Optional[Sequence[RgbTuple]],
    verbose: bool,
) -> Dict[str, object]:
    input_pcd = make_point_cloud(points, colors)
    output_dir = output_root / "full"
    result: Dict[str, object] = {
        "output_dir": str(output_dir.resolve()),
        "method": method,
        "num_points": int(points.shape[0]),
        "status": "pending",
    }

    try:
        trunk_strategy = None
        if method == "slbc":
            if trunk_colors:
                chosen_trunk_colors = list(trunk_colors)
                trunk_strategy = "user_provided"
            else:
                chosen_trunk_colors = infer_trunk_colors(color_groups)
                trunk_strategy = "largest_color_group"
            skeletonizer = run_slbc(
                points=points,
                colors=colors,
                trunk_colors=chosen_trunk_colors,
                down_sample=down_sample,
                verbose=verbose,
            )
            result["trunk_colors"] = [list(color) for color in chosen_trunk_colors]
            result["trunk_strategy"] = trunk_strategy
        else:
            skeletonizer = run_lbc(input_pcd, down_sample=down_sample, verbose=verbose)

        export_result_bundle(
            output_dir=output_dir,
            input_pcd=input_pcd,
            skeletonizer=skeletonizer,
            metadata={
                **result,
                "status": "ok",
                "input_file": str(input_ply.resolve()),
                "down_sample": down_sample,
            },
        )
        result["status"] = "ok"
    except Exception as exc:  # pragma: no cover - integration behavior
        result["status"] = "failed"
        result["reason"] = str(exc)
        logging.exception("Full-cloud skeletonization failed")

    return result


def write_summary(summary_path: Path, payload: Dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.input_ply.exists():
        raise FileNotFoundError(f"Input PLY not found: {args.input_ply}")

    if args.instance_mode == "external_instance" and args.instance_file is None:
        raise ValueError("--instance-file is required when --instance-mode external_instance")

    points, colors = load_ascii_ply_with_rgb(args.input_ply)

    if args.instance_mode == "color_label":
        groups = group_by_color(colors)
    else:
        assert args.instance_file is not None
        instance_ids = load_external_instance_ids(args.instance_file, expected_points=points.shape[0])
        groups = group_by_external_instances(instance_ids)

    trunk_colors = None
    if args.trunk_colors:
        trunk_colors = [parse_rgb_triplet(value) for value in args.trunk_colors]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_results = skeletonize_instances(
        points=points,
        colors=colors,
        groups=groups,
        output_root=output_dir,
        input_ply=args.input_ply,
        down_sample=args.down_sample,
        min_points=args.min_points,
        verbose=args.verbose,
    )
    full_result = skeletonize_full_cloud(
        points=points,
        colors=colors,
        color_groups=group_by_color(colors),
        output_root=output_dir,
        input_ply=args.input_ply,
        method=args.method,
        down_sample=args.down_sample,
        trunk_colors=trunk_colors,
        verbose=args.verbose,
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_ply": str(args.input_ply.resolve()),
        "output_dir": str(output_dir),
        "instance_mode": args.instance_mode,
        "instance_file": str(args.instance_file.resolve()) if args.instance_file else None,
        "method": args.method,
        "down_sample": args.down_sample,
        "min_points": args.min_points,
        "num_points": int(points.shape[0]),
        "num_instances_discovered": len(groups),
        "num_instances_ok": sum(result["status"] == "ok" for result in instance_results),
        "num_instances_skipped": sum(result["status"] == "skipped" for result in instance_results),
        "num_instances_failed": sum(result["status"] == "failed" for result in instance_results),
        "full": full_result,
        "instances": instance_results,
    }
    write_summary(output_dir / "summary.json", summary)

    print(f"[INFO] Processed {len(groups)} instances from {args.input_ply}")
    print(f"[INFO] Summary written to {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
