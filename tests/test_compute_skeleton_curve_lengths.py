import json
import struct
import sys
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from compute_skeleton_curve_lengths import (  # noqa: E402
    analyze_entry,
    analyze_skeleton_points,
    compute_component_curve_length,
    minimum_spanning_tree,
    read_ply_points,
    tree_diameter_path,
)


def write_binary_point_ply(path: Path, points: np.ndarray) -> None:
    with path.open("wb") as f:
        header = "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "element vertex {}".format(points.shape[0]),
                "property double x",
                "property double y",
                "property double z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
            ]
        )
        f.write(header.encode("utf-8"))
        f.write(b"\n")
        for x, y, z in points:
            f.write(struct.pack("<dddBBB", float(x), float(y), float(z), 0, 0, 255))


def test_read_ply_points_reads_binary_point_cloud(tmp_path):
    ply_path = tmp_path / "skeleton.ply"
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    write_binary_point_ply(ply_path, points)

    loaded = read_ply_points(ply_path)

    np.testing.assert_allclose(loaded, points)


def test_tree_diameter_path_prefers_main_chain_over_short_branch():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.0, 0.4, 0.0],
        ]
    )
    edges = [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (1, 4, 0.4),
    ]

    mst_edges = minimum_spanning_tree([0, 1, 2, 3, 4], edges)
    path, length = tree_diameter_path([0, 1, 2, 3, 4], mst_edges, node_count=len(points))

    assert path == [0, 1, 2, 3]
    assert length == 3.0


def test_compute_component_curve_length_handles_small_branch_tree():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.0, 0.4, 0.0],
        ]
    )
    edges = [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (1, 4, 0.4),
    ]

    result = compute_component_curve_length(
        points=points,
        node_indices=[0, 1, 2, 3, 4],
        component_edges=edges,
        node_count=len(points),
        component_index=0,
        resample_points=50,
        spline_smoothing=0.0,
    )

    assert result.had_cycle_before_cleanup is False
    assert result.polyline_length == 3.0
    assert abs(result.curve_length - 3.0) < 1e-6


def test_analyze_skeleton_points_sums_multiple_components():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    metrics = analyze_skeleton_points(
        points,
        k_neighbors=1,
        resample_points=25,
        spline_smoothing=0.0,
    )

    assert metrics["component_count"] == 2
    assert abs(metrics["curve_length_sum"] - 3.0) < 1e-6
    assert abs(metrics["largest_component_length"] - 2.0) < 1e-6


def test_analyze_entry_marks_missing_or_failed_group(tmp_path):
    entry = {
        "name": "label_32",
        "label": 32,
        "status": "failed",
        "reason": "original skeletonization failed",
    }

    result = analyze_entry(
        entry=entry,
        scope="group",
        skeleton_ply=tmp_path / "missing.ply",
        k_neighbors=4,
        resample_points=50,
        spline_smoothing=0.0,
    )

    assert result["status"] == "missing_or_failed"
    assert "failed" in result["reason"]


def test_analyze_entry_reads_realistic_temp_group(tmp_path):
    group_dir = tmp_path / "label_00"
    group_dir.mkdir()
    skeleton_ply = group_dir / "skeleton.ply"
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    write_binary_point_ply(skeleton_ply, points)
    entry = {"name": "label_00", "label": 0, "status": "ok"}

    result = analyze_entry(
        entry=entry,
        scope="group",
        skeleton_ply=skeleton_ply,
        k_neighbors=1,
        resample_points=20,
        spline_smoothing=0.0,
    )

    assert result["status"] == "ok"
    assert result["component_count"] == 1
    assert abs(result["curve_length_sum"] - 2.0) < 1e-6
