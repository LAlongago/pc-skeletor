import json
import sys
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from skeletonize_pointcept_instances import (  # noqa: E402
    ensure_palette,
    group_by_color,
    group_by_labels,
    load_coord_and_labels,
    load_ascii_ply_with_rgb,
    load_external_instance_ids,
)


def test_load_ascii_ply_with_rgb(tmp_path):
    ply_path = tmp_path / "sample.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0.0 0.1 0.2 255 0 0",
                "1.0 1.1 1.2 0 255 0",
                "2.0 2.1 2.2 0 0 255",
            ]
        ),
        encoding="utf-8",
    )

    points, colors = load_ascii_ply_with_rgb(ply_path)

    assert points.shape == (3, 3)
    assert colors.shape == (3, 3)
    np.testing.assert_allclose(points[1], np.array([1.0, 1.1, 1.2], dtype=np.float32))
    np.testing.assert_array_equal(colors[2], np.array([0, 0, 255], dtype=np.uint8))


def test_group_by_color_returns_stable_color_groups():
    colors = np.array(
        [
            [0, 0, 255],
            [255, 0, 0],
            [0, 0, 255],
            [255, 0, 0],
            [0, 255, 0],
        ],
        dtype=np.uint8,
    )

    groups = group_by_color(colors)

    assert [group["name"] for group in groups] == [
        "color_0_0_255",
        "color_0_255_0",
        "color_255_0_0",
    ]
    assert [group["num_points"] for group in groups] == [2, 1, 2]


def test_group_by_labels_returns_stable_label_groups():
    labels = np.array([5, 3, 5, 1, 3], dtype=np.int64)

    groups = group_by_labels(labels)

    assert [group["name"] for group in groups] == [
        "label_01",
        "label_03",
        "label_05",
    ]
    assert [group["num_points"] for group in groups] == [1, 2, 2]


def test_load_external_instance_ids_from_json(tmp_path):
    json_path = tmp_path / "instance_ids.json"
    json_path.write_text(json.dumps({"instance_ids": [7, 7, 9]}), encoding="utf-8")

    instance_ids = load_external_instance_ids(json_path, expected_points=3)

    np.testing.assert_array_equal(instance_ids, np.array([7, 7, 9], dtype=np.int64))


def test_load_coord_and_labels_generates_palette_colors(tmp_path):
    coord_path = tmp_path / "coord.npy"
    pred_path = tmp_path / "pred.npy"
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    labels = np.array([0, 3], dtype=np.int64)
    np.save(coord_path, coords)
    np.save(pred_path, labels)

    points, loaded_labels, colors = load_coord_and_labels(coord_path, pred_path, num_classes=36)
    palette = ensure_palette(36)

    np.testing.assert_allclose(points, coords)
    np.testing.assert_array_equal(loaded_labels, labels)
    np.testing.assert_array_equal(colors[0], palette[0])
    np.testing.assert_array_equal(colors[1], palette[3])
