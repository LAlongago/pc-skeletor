#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skeletonize Pointcept-exported point clouds with pc-skeletor.

Preferred usage for Pointcept part segmentation is:
    --coord-npy <coord.npy> --pred-npy <sample_pred.npy>

In that mode, the script exports one skeleton per predicted label and also one
full skeleton for the entire point cloud. A legacy PLY mode is kept as a
fallback for color-based grouping.
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

PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
        [128, 128, 128],
        [255, 99, 71],
        [154, 205, 50],
        [30, 144, 255],
        [255, 140, 0],
        [186, 85, 211],
        [0, 206, 209],
        [255, 20, 147],
        [124, 252, 0],
        [255, 182, 193],
        [32, 178, 170],
        [221, 160, 221],
        [160, 82, 45],
        [255, 239, 213],
        [139, 0, 0],
        [127, 255, 212],
        [85, 107, 47],
    ],
    dtype=np.uint8,
)

IGNORE_COLOR = np.array([80, 80, 80], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Skeletonize Pointcept outputs. Preferred mode uses coord.npy and "
            "pred.npy to generate one skeleton per label plus a full skeleton."
        )
    )
    parser.add_argument(
        "--coord-npy",
        type=Path,
        default=None,
        help="Pointcept coord.npy for the sample.",
    )
    parser.add_argument(
        "--pred-npy",
        type=Path,
        default=None,
        help="Pointcept predicted label file, e.g. 8_pred.npy.",
    )
    parser.add_argument(
        "--input-ply",
        type=Path,
        default=None,
        help="Fallback input PLY for color-based grouping.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for skeleton outputs.")
    parser.add_argument(
        "--group-mode",
        type=str,
        choices=("label", "color_label", "external_instance"),
        default="label",
        help="How to split the point cloud into per-group skeletons.",
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
        help="Whole-point-cloud skeletonization method. Per-group skeletons always use LBC.",
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
        help="Skip groups with fewer than this many points.",
    )
    parser.add_argument(
        "--trunk-labels",
        nargs="*",
        type=int,
        default=None,
        help="Optional label ids for full-cloud SLBC when labels are available.",
    )
    parser.add_argument(
        "--trunk-colors",
        nargs="*",
        default=None,
        help="Optional RGB triplets such as 230,25,75 for full-cloud SLBC in color mode.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=36,
        help="Number of semantic classes used for pseudo-color rendering in label mode.",
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


def ensure_palette(num_classes: int) -> np.ndarray:
    if num_classes <= len(PALETTE):
        return PALETTE[:num_classes]
    repeats = (num_classes + len(PALETTE) - 1) // len(PALETTE)
    return np.tile(PALETTE, (repeats, 1))[:num_classes]


def labels_to_color(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    labels = labels.reshape(-1).astype(np.int64)
    colors = np.empty((labels.shape[0], 3), dtype=np.uint8)
    ignore_mask = (labels < 0) | (labels >= len(palette))
    if np.any(~ignore_mask):
        colors[~ignore_mask] = palette[labels[~ignore_mask]]
    if np.any(ignore_mask):
        colors[ignore_mask] = IGNORE_COLOR
    return colors


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


def load_coord_and_labels(coord_path: Path, pred_path: Path, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.load(coord_path)
    labels = np.load(pred_path).reshape(-1).astype(np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"coord.npy must have shape (N, 3), got: {points.shape}")
    if points.shape[0] != labels.shape[0]:
        raise ValueError(
            f"coord.npy and pred.npy size mismatch: {points.shape[0]} vs {labels.shape[0]}"
        )
    palette = ensure_palette(max(num_classes, int(labels.max()) + 1 if labels.size else num_classes))
    colors = labels_to_color(labels, palette)
    return points.astype(np.float32), labels, colors


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


def group_by_labels(labels: np.ndarray) -> List[Dict[str, object]]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    groups: List[Dict[str, object]] = []
    for label, count in sorted(zip(unique_labels.tolist(), counts.tolist()), key=lambda item: item[0]):
        indices = np.flatnonzero(labels == label)
        groups.append(
            {
                "name": f"label_{int(label):02d}",
                "label": int(label),
                "indices": indices,
                "num_points": int(count),
            }
        )
    return groups


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


def run_slbc(
    points: np.ndarray,
    colors: np.ndarray,
    trunk_mask: np.ndarray,
    down_sample: float,
    verbose: bool = False,
):
    from pc_skeletor import SLBC

    if trunk_mask.shape[0] != points.shape[0]:
        raise ValueError("SLBC trunk mask length does not match point count.")
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


def infer_largest_group(groups: Sequence[Dict[str, object]], key_name: str) -> object:
    largest_group = max(groups, key=lambda group: int(group["num_points"]))
    return largest_group[key_name]


def summarize_group(group: Dict[str, object]) -> Dict[str, object]:
    summary = {
        "name": group["name"],
        "num_points": int(group["num_points"]),
    }
    if "label" in group:
        summary["label"] = int(group["label"])
    if "color" in group:
        summary["color"] = list(group["color"])  # type: ignore[arg-type]
    if "instance_id" in group:
        summary["instance_id"] = int(group["instance_id"])
    return summary


def skeletonize_groups(
    points: np.ndarray,
    colors: np.ndarray,
    groups: Sequence[Dict[str, object]],
    output_root: Path,
    input_reference: Path,
    down_sample: float,
    min_points: int,
    verbose: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    groups_root = output_root / "groups"
    groups_root.mkdir(parents=True, exist_ok=True)

    for group in groups:
        result = summarize_group(group)
        indices = group["indices"]  # type: ignore[assignment]
        group_points = points[indices]
        group_colors = colors[indices]
        result["output_dir"] = str((groups_root / str(group["name"])).resolve())
        result["method"] = "lbc"

        if int(group["num_points"]) < min_points:
            result["status"] = "skipped"
            result["reason"] = f"num_points < min_points ({group['num_points']} < {min_points})"
            results.append(result)
            continue

        try:
            group_pcd = make_point_cloud(group_points, group_colors)
            skeletonizer = run_lbc(group_pcd, down_sample=down_sample, verbose=verbose)
            output_dir = groups_root / str(group["name"])
            export_result_bundle(
                output_dir=output_dir,
                input_pcd=group_pcd,
                skeletonizer=skeletonizer,
                metadata={
                    **result,
                    "status": "ok",
                    "input_file": str(input_reference.resolve()),
                    "down_sample": down_sample,
                    "min_points": min_points,
                },
            )
            result["status"] = "ok"
        except Exception as exc:  # pragma: no cover - integration behavior
            result["status"] = "failed"
            result["reason"] = str(exc)
            logging.exception("Group skeletonization failed for %s", group["name"])

        results.append(result)

    return results


def skeletonize_full_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    output_root: Path,
    input_reference: Path,
    method: str,
    down_sample: float,
    trunk_mask: Optional[np.ndarray],
    trunk_info: Optional[Dict[str, object]],
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
        if method == "slbc":
            if trunk_mask is None:
                raise ValueError("SLBC requires trunk selection information.")
            skeletonizer = run_slbc(
                points=points,
                colors=colors,
                trunk_mask=trunk_mask,
                down_sample=down_sample,
                verbose=verbose,
            )
            if trunk_info:
                result.update(trunk_info)
        else:
            skeletonizer = run_lbc(input_pcd, down_sample=down_sample, verbose=verbose)

        export_result_bundle(
            output_dir=output_dir,
            input_pcd=input_pcd,
            skeletonizer=skeletonizer,
            metadata={
                **result,
                "status": "ok",
                "input_file": str(input_reference.resolve()),
                "down_sample": down_sample,
            },
        )
        result["status"] = "ok"
    except Exception as exc:  # pragma: no cover - integration behavior
        result["status"] = "failed"
        result["reason"] = str(exc)
        logging.exception("Full-cloud skeletonization failed")

    return result


def build_trunk_selection(
    method: str,
    group_mode: str,
    groups: Sequence[Dict[str, object]],
    labels: Optional[np.ndarray],
    colors: np.ndarray,
    trunk_labels: Optional[Sequence[int]],
    trunk_colors: Optional[Sequence[RgbTuple]],
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, object]]]:
    if method != "slbc":
        return None, None

    if group_mode == "label" and labels is not None:
        if trunk_labels:
            chosen_labels = sorted(set(int(label) for label in trunk_labels))
            mask = np.isin(labels, np.asarray(chosen_labels, dtype=np.int64))
            info = {"trunk_labels": chosen_labels, "trunk_strategy": "user_provided_labels"}
        else:
            largest_label = int(infer_largest_group(groups, "label"))
            mask = labels == largest_label
            info = {"trunk_labels": [largest_label], "trunk_strategy": "largest_label_group"}
        return mask, info

    if trunk_colors:
        chosen_colors = [parse_rgb_triplet(value) if isinstance(value, str) else value for value in trunk_colors]
        color_array = np.asarray(chosen_colors, dtype=np.uint8)
        mask = np.any(np.all(colors[:, None, :] == color_array[None, :, :], axis=2), axis=1)
        info = {
            "trunk_colors": [list(color) for color in chosen_colors],
            "trunk_strategy": "user_provided_colors",
        }
        return mask, info

    largest_color = infer_largest_group(group_by_color(colors), "color")
    chosen_color = tuple(int(value) for value in largest_color)  # type: ignore[arg-type]
    color_array = np.asarray(chosen_color, dtype=np.uint8)
    mask = np.all(colors == color_array[None, :], axis=1)
    info = {
        "trunk_colors": [list(chosen_color)],
        "trunk_strategy": "largest_color_group",
    }
    return mask, info


def resolve_inputs(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Path]:
    if args.group_mode == "label":
        if args.coord_npy is None or args.pred_npy is None:
            raise ValueError("--group-mode label requires both --coord-npy and --pred-npy")
        if not args.coord_npy.exists():
            raise FileNotFoundError(f"coord.npy not found: {args.coord_npy}")
        if not args.pred_npy.exists():
            raise FileNotFoundError(f"pred.npy not found: {args.pred_npy}")
        points, labels, colors = load_coord_and_labels(
            coord_path=args.coord_npy,
            pred_path=args.pred_npy,
            num_classes=args.num_classes,
        )
        return points, colors, labels, args.pred_npy

    if args.input_ply is None:
        raise ValueError(
            f"--group-mode {args.group_mode} requires --input-ply unless label mode is used"
        )
    if not args.input_ply.exists():
        raise FileNotFoundError(f"Input PLY not found: {args.input_ply}")
    points, colors = load_ascii_ply_with_rgb(args.input_ply)
    return points, colors, None, args.input_ply


def write_summary(summary_path: Path, payload: Dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.group_mode == "external_instance" and args.instance_file is None:
        raise ValueError("--instance-file is required when --group-mode external_instance")

    points, colors, labels, input_reference = resolve_inputs(args)

    if args.group_mode == "label":
        assert labels is not None
        groups = group_by_labels(labels)
    elif args.group_mode == "color_label":
        groups = group_by_color(colors)
    else:
        assert args.instance_file is not None
        if not args.instance_file.exists():
            raise FileNotFoundError(f"Instance file not found: {args.instance_file}")
        instance_ids = load_external_instance_ids(args.instance_file, expected_points=points.shape[0])
        groups = group_by_external_instances(instance_ids)

    parsed_trunk_colors = None
    if args.trunk_colors:
        parsed_trunk_colors = [parse_rgb_triplet(value) for value in args.trunk_colors]

    trunk_mask, trunk_info = build_trunk_selection(
        method=args.method,
        group_mode=args.group_mode,
        groups=groups,
        labels=labels,
        colors=colors,
        trunk_labels=args.trunk_labels,
        trunk_colors=parsed_trunk_colors,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    group_results = skeletonize_groups(
        points=points,
        colors=colors,
        groups=groups,
        output_root=output_dir,
        input_reference=input_reference,
        down_sample=args.down_sample,
        min_points=args.min_points,
        verbose=args.verbose,
    )
    full_result = skeletonize_full_cloud(
        points=points,
        colors=colors,
        output_root=output_dir,
        input_reference=input_reference,
        method=args.method,
        down_sample=args.down_sample,
        trunk_mask=trunk_mask,
        trunk_info=trunk_info,
        verbose=args.verbose,
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_reference": str(input_reference.resolve()),
        "coord_npy": str(args.coord_npy.resolve()) if args.coord_npy else None,
        "pred_npy": str(args.pred_npy.resolve()) if args.pred_npy else None,
        "input_ply": str(args.input_ply.resolve()) if args.input_ply else None,
        "output_dir": str(output_dir),
        "group_mode": args.group_mode,
        "instance_file": str(args.instance_file.resolve()) if args.instance_file else None,
        "method": args.method,
        "down_sample": args.down_sample,
        "min_points": args.min_points,
        "num_points": int(points.shape[0]),
        "num_groups_discovered": len(groups),
        "num_groups_ok": sum(result["status"] == "ok" for result in group_results),
        "num_groups_skipped": sum(result["status"] == "skipped" for result in group_results),
        "num_groups_failed": sum(result["status"] == "failed" for result in group_results),
        "full": full_result,
        "groups": group_results,
    }
    write_summary(output_dir / "summary.json", summary)

    print(f"[INFO] Processed {len(groups)} groups from {input_reference}")
    print(f"[INFO] Summary written to {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
