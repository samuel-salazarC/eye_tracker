"""Record iris positions from a webcam with MediaPipe Face Landmarker and save them to CSV."""

import argparse
import csv
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")

# Centre of each iris in the 478-point face landmark model
LEFT_IRIS = 468
RIGHT_IRIS = 473

CSV_COLUMNS = ["time", "x", "y", "left_x", "left_y", "right_x", "right_y"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="gaze_data.csv",
                        help="CSV file for the recorded positions (default: gaze_data.csv)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Index of the webcam to use (default: 0)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop automatically after this many seconds")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help=f"Path to the Face Landmarker model (default: {DEFAULT_MODEL_PATH})")
    return parser.parse_args()


def ensure_model(path: Path) -> Path:
    """Download the Face Landmarker model the first time the tracker runs."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the Face Landmarker model to {path}...")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def iris_positions(landmarks, width: int, height: int):
    """Return the pixel positions of both irises and of their midpoint.

    ``landmarks`` is the list of normalized landmarks of one face. Returns ``None`` if
    the list does not include the iris points.
    """
    if len(landmarks) <= RIGHT_IRIS:
        return None
    left = (round(landmarks[LEFT_IRIS].x * width), round(landmarks[LEFT_IRIS].y * height))
    right = (round(landmarks[RIGHT_IRIS].x * width), round(landmarks[RIGHT_IRIS].y * height))
    centre = (round((left[0] + right[0]) / 2), round((left[1] + right[1]) / 2))
    return {"left": left, "right": right, "centre": centre}


def main():
    args = parse_args()
    model_path = ensure_model(args.model)

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}.")

    samples = 0
    start = time.monotonic()
    last_timestamp_ms = -1

    with vision.FaceLandmarker.create_from_options(options) as landmarker, \
            open(args.output, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)
        print("Recording. Press 'q' in the video window to stop.")

        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the camera.")
                break

            elapsed = time.monotonic() - start
            if args.duration is not None and elapsed >= args.duration:
                break

            # The video mode requires strictly increasing timestamps
            timestamp_ms = max(int(elapsed * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, timestamp_ms)

            if result.face_landmarks:
                height, width = frame.shape[:2]
                points = iris_positions(result.face_landmarks[0], width, height)
                if points:
                    writer.writerow([round(elapsed, 3), *points["centre"],
                                     *points["left"], *points["right"]])
                    samples += 1
                    cv2.circle(frame, points["left"], 3, (255, 0, 0), -1)
                    cv2.circle(frame, points["right"], 3, (0, 255, 0), -1)

            cv2.imshow("Eye Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()
    print(f"Saved {samples} samples to {args.output}.")


if __name__ == "__main__":
    main()
