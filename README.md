# Eye Tracker

Webcam-based eye tracking in Python. The tool detects the irises with MediaPipe Face Landmarker, records their position over time in a CSV file and draws a heatmap of where they concentrated.

## Features

- Real-time iris detection from any webcam, with a live preview.
- One CSV row per frame with the position of each iris and their midpoint.
- Heatmap of the recorded positions, displayed on screen or saved as an image.
- Command-line options for the camera, the output file and the recording time.

## How it works

1. `eye_tracker.py` reads frames from the webcam and runs the MediaPipe Face Landmarker model, which returns 478 face landmarks, including the centre of each iris.
2. For every frame with a detected face, it saves the time and the pixel coordinates of both irises to a CSV file.
3. `heatmap.py` estimates the density of those positions (kernel density estimation with Seaborn) and plots it.

The coordinates describe where the irises appear in the camera image. They are not calibrated to the screen, so the heatmap shows how the eyes moved, not exactly which point of the screen was being looked at.

## Tech stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| MediaPipe (Face Landmarker) | Face and iris landmarks |
| OpenCV | Webcam capture and preview |
| pandas | Data loading |
| Matplotlib, Seaborn | Heatmap |
| pytest | Unit tests |

## Setup

```bash
git clone https://github.com/samuel-salazarC/eye_tracker.git
cd eye_tracker

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the Face Landmarker model (about 4 MB) to `models/face_landmarker.task`.

On macOS, the terminal needs permission to use the camera (System Settings, Privacy & Security, Camera).

## Usage

Record the iris positions. Press `q` in the video window to stop.

```bash
python eye_tracker.py
```

| Option | Description |
|--------|-------------|
| `-o`, `--output` | CSV file for the recorded positions. Defaults to `gaze_data.csv`. |
| `--camera` | Index of the webcam. Defaults to `0`. |
| `--duration` | Stops automatically after this many seconds. |
| `--model` | Path to the Face Landmarker model. Defaults to `models/face_landmarker.task`. |

Draw the heatmap:

```bash
python heatmap.py                         # opens a window
python heatmap.py -o heatmap.png          # saves the image
python heatmap.py -i session.csv -o session.png
```

### CSV format

| Column | Description |
|--------|-------------|
| `time` | Seconds since the recording started |
| `x`, `y` | Midpoint between both irises, in pixels |
| `left_x`, `left_y` | Left iris, in pixels |
| `right_x`, `right_y` | Right iris, in pixels |

## Tests

```bash
pytest
```

The tests cover the conversion of landmarks to pixel coordinates and the heatmap generation. They do not need a webcam.

## Project structure

```
eye_tracker/
├── eye_tracker.py          # Webcam capture and CSV recording
├── heatmap.py              # Heatmap of the recorded positions
├── requirements.txt
└── tests/
    └── test_eye_tracker.py
```

## Possible improvements

- Calibrate the iris positions against known points on the screen to estimate the real gaze point.
- Overlay the heatmap on a screenshot of the interface being studied.

## Author

Samuel Salazar, Data Science Engineering student. [GitHub profile](https://github.com/samuel-salazarC)
