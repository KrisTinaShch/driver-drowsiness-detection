# Driver Drowsiness Detection

A webcam watches the driver's eyes. A small neural network decides, frame by
frame, whether an eye is open or closed; a timer turns that into a decision —
if the eyes stay shut for two seconds, an alarm sounds.

Trained on the [MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset):
84,898 images of 37 people. Measured on eight people the model never saw:
**97.5% accuracy**, against a 48.7% majority baseline.

## How it works

```
webcam frame
  ├─ MediaPipe         finds the face and 478 landmarks
  ├─ eye crop          a square around the eyelid contour, 64x64, grayscale
  ├─ ResNet-18         probability that the eye is closed
  └─ ClosureTracker    smoothing, a two-second timer, the alarm
```

Only the classifier was trained. Face detection comes ready-made from
MediaPipe and the eye boxes are computed from its landmarks: the dataset holds
cropped eyes with a state label and no bounding boxes, so no detector could be
trained on it.

## The split

The dataset ships with train/val/test folders, but all 37 subjects appear in
all three — the split was made frame-wise, and consecutive frames of one
recording are nearly identical. A model scores high by recognising pictures it
has already seen.

Splitting by subject alone is not enough either. The share of closed-eye frames
per person runs from 2% to 100%, so a random shuffle of people left 83% closed
frames in the test set, where answering "closed" every time would score 83%.

The split used assigns people one at a time, largest first, each to whichever
subset has the smallest combined penalty for overshooting its quota and for
skewing its class balance. It is deterministic — no random seed, so it rebuilds
identically from the code.

| subset | images | people | closed |
|---|---|---|---|
| train | 51,430 | 22 | 0.498 |
| val | 16,734 | 7 | 0.490 |
| test | 16,734 | 8 | 0.487 |

## Model

ResNet-18 pretrained on ImageNet, the first convolution summed down to one
channel, single-logit head. Input 64x64 grayscale. `BCEWithLogitsLoss`, AdamW
at `3e-4`, batch 256, 5 epochs — about four minutes on an RTX 3070 Ti.
Augmentation: horizontal flip, gamma, brightness and contrast jitter, Gaussian
noise, light blur.

Label 1 means closed, so recall answers the question that matters: what share
of real closures were caught.

## Results

| | |
|---|---|
| accuracy | 0.9750 |
| ROC-AUC | 0.996 |
| recall on closed eyes | 0.9707 |
| majority baseline | 0.487 |

180 false alarms and 239 misses out of 16,734 frames — close to symmetric. A
single missed frame never reaches the alarm: that needs two seconds, about
sixty frames in a row.

Weakest cases are glasses (0.9567 against 0.9819 without) and strong
reflections (0.9466). Lighting barely matters, which is what the gamma and
brightness augmentation was for.

## What the experiments showed

`experiments.ipynb` compares eight architectures and every other choice under
one split and one augmentation.

**Pretraining is the lever, not size.** The same ResNet-18 scores 0.9774 with
ImageNet weights and 0.9569 from scratch. The architecture itself is worth
less: from scratch it beats a plain 6-layer CNN by 0.0124, despite having
eighty times the parameters.

**Augmentation is not optional.** Without it, accuracy on noisy frames falls to
0.6355; with the full set, 0.9613. Each distortion only cures the one it
resembles — noise fixes noise, gamma and brightness fix darkness.

**Differences under 0.0075 mean nothing.** That is how far one unchanged model
moves between epochs, so smaller gaps are not evidence.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux
pip install -r requirements.txt
python download_model.py        # fetches the MediaPipe face landmark model
```

## Running

**Train.** Open `notebook.ipynb` and run it top to bottom. It downloads the
dataset through `kagglehub` (no Kaggle token needed — the dataset is public),
builds the manifest, trains, and writes `best.pt`.

**Demo.** With `best.pt` in place:

```bash
python demo.py
```

Quit with `q`, `Esc`, or by closing the window.

**Tests.**

```bash
pytest -q
```

Ten tests drive `ClosureTracker` with a fake clock: a blink of 0.15 s, a
closure of 1.5 s that must stay silent, one of 2.5 s that must not, a face
disappearing mid-closure. None of that is reproducible in front of a camera.

## Files

| | |
|---|---|
| `notebook.ipynb` | data, split, training, evaluation |
| `experiments.ipynb` | the comparisons behind every choice |
| `demo.py` | webcam loop |
| `tracker.py` | smoothing and the closure timer |
| `test_tracker.py` | its tests |
| `alarm.py` | the sound |
| `download_model.py` | fetches the MediaPipe model |

Weights, the manifest and the dataset are not committed: all three are
reproducible from the code.

## Limitations

- Eight test subjects show a six-point spread in accuracy and are too few to
  bound it.
- Darkness was simulated, not filmed. Night and in-car conditions are untested.
- Some of the remaining errors belong to the dataset: among the most confident
  mistakes are runs of consecutive frames labelled "open" on which every model
  trained here answers "closed".
