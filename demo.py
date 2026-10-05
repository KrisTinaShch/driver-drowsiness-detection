import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import torch
import torch.nn as nn
import torchvision
from alarm import Alarm
from tracker import ClosureTracker

WINDOW = "Drowsiness detector"
FACE_MODEL = "models/face_landmarker.task"
EYE_MODEL = "best.pt"

CROP = 96
IMG = 64
THRESHOLD = 0.5
ALARM_AFTER = 2.0
LOST_WARN = 1.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

GREEN, RED, WHITE, GREY = (0, 200, 0), (0, 0, 255), (255, 255, 255), (90, 90, 90)

FONT = cv2.FONT_HERSHEY_SIMPLEX

def text_scale(size):
    return size / 30


def draw_texts(frame, items):
    for text, (x, y), size, color in items:
        scale = text_scale(size)
        (_, height), _ = cv2.getTextSize(text, FONT, scale, 2)
        cv2.putText(frame, text, (x, y + height), FONT, scale, color, 2, cv2.LINE_AA)
    return frame


def eye_box(landmarks, ids, w, h, margin=0.6):
    xs = np.array([landmarks[i].x for i in ids]) * w
    ys = np.array([landmarks[i].y for i in ids]) * h

    cx, cy = xs.mean(), ys.mean()
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + margin) / 2

    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


def load_eye_model():
    model = torchvision.models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load(EYE_MODEL, map_location="cpu"))
    return model.eval().to(DEVICE)


@torch.no_grad()
def closed_prob(model, eyes):
    batch = np.stack([cv2.resize(e, (IMG, IMG)) for e in eyes])
    x = torch.from_numpy(batch).float().div(255).sub(0.5).div(0.5)
    x = x.unsqueeze(1).to(DEVICE)
    return torch.sigmoid(model(x).squeeze(1)).cpu().numpy()


def draw_state(frame, state, texts):
    h, w = frame.shape[:2]
    color = RED if state["closed"] else GREEN

    texts.append(("eyes closed" if state["closed"] else "eyes open",(10, 14 + CROP), 26, color))

    filled = min(state["duration"] / ALARM_AFTER, 1.0)
    x1, y1, x2, y2 = 10, h - 58, w - 10, h - 38
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREY, 1)
    cv2.rectangle(frame, (x1, y1), (x1 + int((x2 - x1) * filled), y2), color, -1)
    texts.append((f"{state['duration']:.1f} of {ALARM_AFTER:.0f} s", (x1, y1 - 30), 22, WHITE))

    if state["lost"] > LOST_WARN:
        (width, _), _ = cv2.getTextSize("no face", FONT, text_scale(22), 2)
        texts.append(("no face", (w - 10 - width, h - 34), 22, RED))

    if state["alarm"]:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 6)
        (width, _), _ = cv2.getTextSize("WAKE UP!", FONT, text_scale(64), 2)
        texts.append(("WAKE UP!", ((w - width) // 2, h // 2 - 40), 64, RED))


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

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cam = cv2.VideoCapture(0, backend)
    if not cam.isOpened():
        alarm.close()
        raise SystemExit("Could not open the camera.")

    start = time.time()
    fps, last = 0.0, time.time()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
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
                    probs = closed_prob(eye_model, eyes)

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
            cv2.imshow(WINDOW, frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        alarm.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
