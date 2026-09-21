# Driver Drowsiness Detection

Real-time detection of driver drowsiness from a webcam. A CNN classifies each
eye as open or closed; a tracking layer measures how long the eyes stay shut and
raises an audible alarm after two seconds.

Trained on the [MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset)
(84,898 images, 37 subjects). Evaluated on people the model has never seen:
**97.0% accuracy, ROC-AUC 0.995**.

## How it works

```
webcam frame
    │
    ├─ MediaPipe FaceLandmarker ──► 478 face landmarks ──► two square eye crops
    │
    ├─ ResNet-18 (grayscale, 64x64) ──► P(closed) for each eye
    │
    ├─ ClosureTracker ──► exponential smoothing ──► threshold ──► closure timer
    │                                                         └─► PERCLOS
    └─ Alarm ──► beep after 2 s of uninterrupted closure
```

The two models do different jobs. MediaPipe answers *where the eyes are*; the
trained classifier answers *what state they are in*. The classifier alone cannot
locate an eye in a 640x480 frame — it was never trained to.

## The split problem

This turned out to be the most consequential part of the project.

### The provided split leaks

The dataset ships with `train` / `val` / `test` folders, but all 37 subjects
appear in all three. The split was made frame-wise: consecutive frames of the
same recording — nearly identical images — end up on both sides of the divide.
A model can score high by recognising a frame it has already memorised.

Our actual use case is a driver the model has never seen, so the provided split
measures the wrong thing. We re-split by subject.

### A plain group split is not enough

Subjects differ wildly. Frame counts range from 382 to 10,257, and the share of
closed-eye frames per subject ranges from **2% to 100%**. Shuffling people at
random keeps subjects intact but destroys class balance:

| split method | train closed | val closed | test closed |
|---|---|---|---|
| random shuffle of subjects | 0.464 | 0.394 | **0.834** |
| `StratifiedGroupKFold` | 0.472 | 0.393 | **0.604** |
| greedy assignment (used here) | 0.498 | 0.490 | **0.487** |

With 83% closed frames in the test set, a model that always answers "closed"
scores 83%, and the metric stops meaning anything. `StratifiedGroupKFold` helps
but does not converge with only 37 highly heterogeneous groups.

### The split used

Subjects are sorted largest first and assigned greedily: each one goes to the
subset where the sum of two penalties is smallest — how far past its quota the
subset would go, and how skewed its class balance would get.

| subset | images | share | subjects | closed |
|---|---|---|---|---|
| train | 51,430 | 0.606 | 22 | 0.498 |
| val | 16,734 | 0.197 | 7 | 0.490 |
| test | 16,734 | 0.197 | 8 | 0.487 |

No subject appears in more than one subset. The assignment is deterministic — no
random seed — so the split is reproducible from the code alone, which is why
`manifest.csv` is not committed.

## Model

ResNet-18 pretrained on ImageNet, adapted to grayscale by summing the RGB
weights of the first convolution, single-logit head. Input 64x64, close to the
dataset's median image size of 87x87. Trained with `BCEWithLogitsLoss`, AdamW at
`3e-4`, mixed precision, batch 256, 5 epochs — about four minutes on an
RTX 3070 Ti. Training-time augmentation: horizontal flip, gamma, brightness and
contrast jitter, Gaussian noise and light blur (see *Robustness* below).

Label convention: **1 = closed**. The positive class is the event being
detected, so recall answers the question that matters — what share of real
closures were caught.

## Results

### Held-out subjects

| | precision | recall | f1 | support |
|---|---|---|---|---|
| open | 0.9719 | 0.9688 | 0.9704 | 8,581 |
| closed | 0.9672 | 0.9706 | 0.9689 | 8,153 |
| **accuracy** | | | **0.9696** | 16,734 |

ROC-AUC 0.9953. Confusion matrix `[[8313, 268], [240, 7913]]` — errors are
close to symmetric. The majority-class baseline is 0.487.

### Where it fails

| condition | accuracy |
|---|---|
| no glasses | 0.9797 |
| glasses | 0.9432 |
| good lighting | 0.9752 |
| poor lighting | 0.9664 |
| no reflections | 0.9787 |
| weak reflections | 0.9690 |
| strong reflections | 0.9050 |
| worst individual subject | 0.9393 |

Glasses and specular reflections are the weak spots, which is what you would
expect: both obscure the eye itself. This is also where the noise augmentation
described below costs the most — those are exactly the cases decided by fine
texture, and the augmentation trains the model not to rely on it.

### The decision threshold barely matters

| threshold | accuracy | recall (closed) | false alarms |
|---|---|---|---|
| 0.5 | 0.9696 | 0.9706 | 268 |
| 0.477 (best F1 on val) | 0.9696 | 0.9712 | 273 |
| 0.249 (recall 0.99 on val) | 0.9685 | 0.9774 | 343 |

With ROC-AUC 0.995 the predicted probabilities pile up near 0 and 1, so moving
the threshold reshuffles only a few hundred frames out of 16,734. Per-frame
tuning is not the lever here — the temporal logic is. A blink flips a single
frame; the alarm needs two seconds of agreement.

### Robustness to noise and low light

The first model was trained with brightness and contrast jitter only. It turned
out to be fragile: Gaussian noise of sigma 4 — barely visible — cost 12 points.
Low light is really a *noise* problem, because a camera raises its gain in the
dark.

Gamma (0.5–2.0), wider brightness jitter, Gaussian noise (sigma up to 14) and
light blur fix it. This is the configuration the notebook ships, and every
number above was produced by it. The comparison below trained one model per
configuration and evaluated both on a 6,000-image sample of the test set:

| condition | brightness/contrast only | full augmentation |
|---|---|---|
| clean frames | 0.9755 | 0.9710 |
| noise sigma 2 | 0.9453 | 0.9713 |
| noise sigma 4 | 0.8582 | 0.9673 |
| noise sigma 8 | 0.6518 | 0.9630 |
| noise sigma 16 | 0.5092 | 0.9467 |
| brightness x0.5 | 0.7695 | 0.9608 |
| brightness x0.3 | 0.6275 | 0.9428 |
| brightness x0.15 | 0.5073 | 0.8632 |
| brightness x0.3 + auto-gain | 0.6062 | 0.9590 |
| brightness x0.15 + auto-gain | 0.4950 | 0.9335 |

Preprocessing the input with CLAHE was tried and made things **worse**
(0.9755 → 0.6355), because the model was trained on unprocessed crops. Input
normalisation only helps if the same normalisation was applied during training.

## Runtime

Measured on an RTX 3070 Ti with a 640x480 webcam at 30 FPS:

| stage | time per frame |
|---|---|
| MediaPipe face landmarks (CPU) | ~25 ms |
| classifier, both eyes in one batch | 3.8 ms |
| label rendering | 1.6 ms |

The classifier is not the bottleneck, so no ONNX export or quantisation was
needed.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux
pip install -r requirements.txt
python download_model.py        # fetches the MediaPipe face landmark model
```

`requirements.txt` is unpinned on purpose. The project started on macOS/Intel
with Python 3.9, and those pins have no prebuilt wheels on other platforms, so
versions are resolved per machine.

## Running

**Train.** Open `notebook.ipynb` and run it top to bottom. It downloads the
dataset through `kagglehub` (no Kaggle token needed — the dataset is public),
builds the manifest, trains, and writes `best.pt`.

By default `kagglehub` caches the dataset in `~/.cache`. To put it elsewhere,
set `KAGGLEHUB_CACHE` in the first cell — it is about 500 MB.

Every training run is logged to MLflow — parameters, the loss and accuracy
curves, test metrics, the per-condition breakdown and the weights themselves.
The log lives in a local SQLite file, so nothing needs to be served:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Demo.** With `best.pt` in place:

```bash
python demo.py
```

Press `q` in the video window to quit. On screen: the two eye crops being fed to
the model, per-eye probabilities, a bar filling toward the two-second threshold,
and PERCLOS over the last minute.

## What is in the repository

| file | |
|---|---|
| `notebook.ipynb` | data exploration, split construction, training, evaluation |
| `demo.py` | webcam loop: landmarks, crops, inference, drawing |
| `tracker.py` | smoothing, closure timer, PERCLOS — no camera, unit-testable |
| `alarm.py` | audible alarm on a persistent output stream |
| `download_model.py` | fetches the MediaPipe model |

Run history (`mlflow.db`, `mlruns/`) is local and not committed either.

Weights (`*.pt`), the manifest and the dataset are not committed: all three are
reproducible from the code.

## Design notes

**PERCLOS.** Besides the two-second timer the demo reports PERCLOS — the share
of time with eyes closed over the last minute. It catches a different failure
mode: a tiring driver blinks more often and for longer without ever holding the
eyes shut for two full seconds. PERCLOS sees that; the timer does not.

**Losing the face does not reset the timer.** When someone falls asleep the head
drops and the face leaves the frame at exactly the moment the alarm matters
most, so the closure counter keeps running.

**Time is measured with a clock, not in frames.** FPS fluctuates, so counting
60 frames would make "two seconds" drift between 1.7 and 2.4.

**The audio stream stays open.** A Bluetooth speaker sleeps during silence and
swallows the first fraction of a second of a sound started on demand.

## Limitations

- Trained on infrared-style indoor captures. It transfers to a visible-light
  webcam well in practice, but this was verified on one person and one camera.
- Squinting and downward gaze sit genuinely between the two classes — the
  dataset has no "half-closed" label. The temporal logic absorbs this, per-frame
  predictions do not.
- Not containerised: the demo needs a camera and a GPU, neither of which passes
  cleanly into a container on Windows. In a production setting this would run on
  a dedicated in-car unit with its own image, most likely with near-infrared
  illumination so that performance does not depend on the time of day.

## Data

MRL Eye Dataset, downloaded via `kagglehub` at runtime. Please refer to the
[dataset page](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset)
for its terms of use.
