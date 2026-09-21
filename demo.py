"""Driver drowsiness detector — real-time demo on a webcam.

MediaPipe locates the eyes, the trained classifier decides open/closed,
ClosureTracker measures how long the eyes stay shut, Alarm sounds after
two seconds.

Run:  python demo.py        Quit:  press q in the video window.
"""
import time

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks.python import BaseOptions, vision

from alarm import Alarm
from tracker import ClosureTracker

FACE_MODEL = "models/face_landmarker.task"
EYE_MODEL = "best.pt"
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"

CROP = 96           # size of the eye preview drawn on screen
IMG = 64            # model input size — same as during training
THRESHOLD = 0.5
ALARM_AFTER = 2.0   # seconds of uninterrupted closure before the alarm
LOST_WARN = 1.5     # seconds without a face before warning

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Landmark indices along each eye contour in the FaceMesh layout (478 points).
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

GREEN, RED, WHITE, GREY = (0, 200, 0), (0, 0, 255), (255, 255, 255), (90, 90, 90)

_fonts = {}


def font(size):
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(FONT_PATH, size)
    return _fonts[size]


def draw_texts(frame, items):
    """Draw labels through PIL: OpenCV cannot render Cyrillic.

    items is a list of (text, (x, y), size, BGR color). One pass per frame:
    converting the frame to PIL and back costs time, doing it per label is waste.
    """
    if not items:
        return frame
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for text, xy, size, color in items:
        draw.text(xy, text, font=font(size), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def eye_box(landmarks, ids, w, h, margin=0.6):
    """Square around an eye from its landmarks. margin is padding relative to eye size."""
    xs = np.array([landmarks[i].x for i in ids]) * w
    ys = np.array([landmarks[i].y for i in ids]) * h

    cx, cy = xs.mean(), ys.mean()
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + margin) / 2

    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def crop(frame, box):
    """Cut out the square without running past the frame edges."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


def load_eye_model():
    """Same architecture as during training, plus the saved weights."""
    model = torchvision.models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load(EYE_MODEL, map_location="cpu"))
    return model.eval().to(DEVICE)


@torch.no_grad()
def closed_prob(model, eyes):
    """eyes is a list of grayscale squares. Returns P(closed) for each.

    Preprocessing must repeat training exactly, otherwise the model sees
    something other than what it learned on: grayscale, 64x64, values mapped
    from [0, 255] to [-1, 1].
    """
    batch = np.stack([cv2.resize(e, (IMG, IMG)) for e in eyes])
    x = torch.from_numpy(batch).float().div(255).sub(0.5).div(0.5)
    x = x.unsqueeze(1).to(DEVICE)                      # (N, 1, 64, 64)
    return torch.sigmoid(model(x).squeeze(1)).cpu().numpy()


def draw_state(frame, state, texts):
    """Closure progress bar, drowsiness measure, alarm banner."""
    h, w = frame.shape[:2]
    color = RED if state["closed"] else GREEN

    texts.append(("глаза закрыты" if state["closed"] else "глаза открыты",
                  (10, 14 + CROP), 26, color))

    # bar: how much of the two seconds has accumulated
    filled = min(state["duration"] / ALARM_AFTER, 1.0)
    x1, y1, x2, y2 = 10, h - 58, w - 10, h - 38
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREY, 1)
    cv2.rectangle(frame, (x1, y1), (x1 + int((x2 - x1) * filled), y2), color, -1)
    texts.append((f"{state['duration']:.1f} из {ALARM_AFTER:.0f} с", (x1, y1 - 30), 22, WHITE))

    # PERCLOS — share of time with eyes closed over the last minute
    perclos = state["perclos"] * 100
    texts.append((f"PERCLOS {perclos:4.1f}%", (x1, h - 34), 22,
                  RED if perclos > 15 else WHITE))

    if state["lost"] > LOST_WARN:
        texts.append(("лицо не видно", (w - 210, h - 34), 22, RED))

    if state["alarm"]:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 6)
        texts.append(("ПРОСНИСЬ!", (w // 2 - 150, h // 2 - 40), 64, RED))


def main():
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    eye_model = load_eye_model()
    tracker = ClosureTracker(threshold=THRESHOLD, alarm_after=ALARM_AFTER)
    alarm = Alarm()
    print("model on", DEVICE)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        alarm.close()
        raise SystemExit("Could not open the camera. Is it busy in another app?")

    start = time.time()
    fps, last = 0.0, time.time()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror it: reads more naturally
            h, w = frame.shape[:2]
            texts = []

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int((time.time() - start) * 1000))

            state = None
            if result.face_landmarks:
                points = result.face_landmarks[0]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                boxes, eyes = [], []
                for ids in (LEFT_EYE, RIGHT_EYE):
                    box = eye_box(points, ids, w, h)
                    eye = crop(gray, box)
                    if eye is not None:
                        boxes.append(box)
                        eyes.append(eye)

                if eyes:
                    probs = closed_prob(eye_model, eyes)   # both eyes in one batch

                    for n, (box, eye, p) in enumerate(zip(boxes, eyes, probs)):
                        color = RED if p > THRESHOLD else GREEN
                        cv2.rectangle(frame, box[:2], box[2:], color, 2)
                        texts.append((f"{p:.2f}", (box[0], box[1] - 26), 22, color))

                        x0 = 10 + n * (CROP + 10)
                        frame[10:10 + CROP, x0:x0 + CROP] = cv2.cvtColor(
                            cv2.resize(eye, (CROP, CROP)), cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(frame, (x0, 10), (x0 + CROP, 10 + CROP), color, 2)

                    state = tracker.update(float(probs.mean()))

            if state is None:
                state = tracker.miss()

            draw_state(frame, state, texts)
            alarm.set(state["alarm"])

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            texts.append((f"FPS {fps:4.1f}", (w - 120, 10), 24, WHITE))

            frame = draw_texts(frame, texts)
            cv2.imshow("Drowsiness detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        alarm.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
