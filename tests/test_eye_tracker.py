from types import SimpleNamespace

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from eye_tracker import LEFT_IRIS, RIGHT_IRIS, iris_positions  # noqa: E402
from heatmap import load_positions, plot_heatmap  # noqa: E402


def make_landmarks(count=478):
    points = [SimpleNamespace(x=0.0, y=0.0) for _ in range(count)]
    if count > RIGHT_IRIS:
        points[LEFT_IRIS] = SimpleNamespace(x=0.40, y=0.50)
        points[RIGHT_IRIS] = SimpleNamespace(x=0.60, y=0.52)
    return points


def test_iris_positions_are_converted_to_pixels():
    points = iris_positions(make_landmarks(), width=640, height=480)

    assert points["left"] == (256, 240)
    assert points["right"] == (384, 250)
    assert points["centre"] == (320, 245)


def test_iris_positions_without_iris_landmarks_returns_none():
    assert iris_positions(make_landmarks(count=468), width=640, height=480) is None


def test_load_positions_drops_incomplete_rows(tmp_path):
    csv_path = tmp_path / "gaze.csv"
    pd.DataFrame({"time": [0, 1, 2], "x": [10, None, 30], "y": [5, 6, 7]}).to_csv(csv_path, index=False)

    data = load_positions(str(csv_path))

    assert list(data["x"]) == [10, 30]


def test_load_positions_requires_x_and_y(tmp_path):
    csv_path = tmp_path / "gaze.csv"
    pd.DataFrame({"time": [0, 1], "x": [1, 2]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="y"):
        load_positions(str(csv_path))


def test_plot_heatmap_creates_a_figure(tmp_path):
    data = pd.DataFrame({"x": [100, 110, 120, 105, 115], "y": [200, 205, 210, 195, 208]})

    figure = plot_heatmap(data, "Test")
    figure.savefig(tmp_path / "heatmap.png")

    assert (tmp_path / "heatmap.png").exists()
