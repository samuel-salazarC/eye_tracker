"""Draw a heatmap of the iris positions recorded by eye_tracker.py."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", default="gaze_data.csv",
                        help="CSV file created by eye_tracker.py (default: gaze_data.csv)")
    parser.add_argument("-o", "--output", default=None,
                        help="Save the heatmap to this image file instead of opening a window")
    parser.add_argument("--title", default="Iris position heatmap", help="Title of the plot")
    return parser.parse_args()


def load_positions(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    missing = {"x", "y"} - set(data.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing the columns: {', '.join(sorted(missing))}")
    data = data.dropna(subset=["x", "y"])
    if len(data) < 2:
        raise ValueError(f"{csv_path} needs at least two samples to draw a heatmap.")
    return data


def plot_heatmap(data: pd.DataFrame, title: str):
    figure, axis = plt.subplots(figsize=(10, 7))
    sns.kdeplot(x=data["x"], y=data["y"], cmap="inferno", fill=True,
                thresh=0.05, levels=100, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("x (pixels in the camera image)")
    axis.set_ylabel("y (pixels in the camera image)")
    axis.invert_yaxis()  # image coordinates grow downwards
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def main():
    args = parse_args()
    try:
        data = load_positions(args.input)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error))

    figure = plot_heatmap(data, args.title)
    if args.output:
        figure.savefig(args.output, dpi=150)
        print(f"Heatmap saved to {args.output}.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
