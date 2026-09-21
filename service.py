"""HTTP service around the eye-state classifier.

The webcam demo cannot be containerised (no camera or display inside a
container on Windows), but the model itself can be served over HTTP: send an
eye crop, get back the probability that the eye is closed.

Local run:  uvicorn service:app --reload
Docs:       http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from fastapi import FastAPI, File, HTTPException, UploadFile

WEIGHTS = "best.pt"
IMG = 64
THRESHOLD = 0.5
MAX_FILES = 16

state: dict = {}


def load_model():
    """Same architecture as during training, weights from disk, CPU only."""
    model = torchvision.models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return model.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not per request: reading 43 MB of weights and
    # building the network takes about a second, a request must not pay that.
    state["model"] = load_model()
    yield
    state.clear()


app = FastAPI(
    title="Eye state classifier",
    description="Returns the probability that an eye is closed.",
    version="1.0",
    lifespan=lifespan,
)


@torch.no_grad()
def closed_prob(crops: list[np.ndarray]) -> np.ndarray:
    """Preprocessing must repeat training exactly: grayscale, 64x64,
    values mapped from [0, 255] to [-1, 1]."""
    batch = np.stack([cv2.resize(c, (IMG, IMG)) for c in crops])
    x = torch.from_numpy(batch).float().div(255).sub(0.5).div(0.5).unsqueeze(1)
    return torch.sigmoid(state["model"](x).squeeze(1)).numpy()


def decode(raw: bytes, name: str) -> np.ndarray:
    """Bytes of an uploaded file into a grayscale array."""
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(400, f"{name}: not a readable image")
    return image


@app.get("/health")
def health():
    """Liveness check — also what Docker's HEALTHCHECK calls."""
    return {"status": "ok", "model_loaded": "model" in state, "threshold": THRESHOLD}


@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    """One or more eye crops. Returns a probability per image.

    `mean_probability` is there for the two-eye case the demo uses: a single
    verdict for a face, averaged over both eyes.
    """
    if not files:
        raise HTTPException(400, "no files")
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"at most {MAX_FILES} files per request")

    crops = [decode(await f.read(), f.filename or "file") for f in files]
    probs = closed_prob(crops)

    return {
        "predictions": [
            {
                "filename": f.filename,
                "closed_probability": round(float(p), 4),
                "closed": bool(p > THRESHOLD),
            }
            for f, p in zip(files, probs)
        ],
        "mean_probability": round(float(probs.mean()), 4),
        "closed": bool(probs.mean() > THRESHOLD),
    }
