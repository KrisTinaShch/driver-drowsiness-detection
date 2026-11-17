# Driver Drowsiness Detection

A webcam watches the driver's eyes. A small neural network decides, frame by
frame, whether an eye is open or closed; a timer turns that into a decision —
if the eyes stay shut for two seconds, an alarm sounds.

Trained on the [MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset):
84,898 images of 37 people. Measured on eight people the model never saw:
**97.5% accuracy**, against a 48.7% majority baseline.

## Quick start

Python 3.10 or 3.11. About five minutes, most of it `pip`.

```bash
git clone https://github.com/KrisTinaShch/driver-drowsiness-detection.git
cd driver-drowsiness-detection
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on Linux
pip install -r requirements-lock.txt
python download_model.py
```

The tests run immediately — no dataset, no weights, no camera:

```bash
pytest -q
```

The demo also needs the trained weights, which are not in the repository:
they are 43 MB and change on every training run. They live in the
[v1.0 release](https://github.com/KrisTinaShch/driver-drowsiness-detection/releases/download/v1.0/best.pt) instead:

```bash
curl -L -o best.pt https://github.com/KrisTinaShch/driver-drowsiness-detection/releases/download/v1.0/best.pt
```

On Windows PowerShell write `curl.exe`, not `curl` — the bare name there is an
alias for something else. Or just download the file in a browser and drop it
into the project folder. Then:

```bash
python demo.py
```

`demo.py` checks both model files before touching the camera and names whatever
is missing, so a wrong setup fails with one line instead of a traceback.

**You also need** a webcam and a working audio output. On Linux `sounddevice`
needs the system package `libportaudio2`, otherwise the alarm stays silent
while everything else works.

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
per person runs from 2% to 100%. Across a thousand random shuffles of people the
test set lands anywhere between 0.201 and 0.783 closed, and a third of them miss
half by more than ten points — whatever such a run scores, answering "closed"
every time scores the same. `StratifiedGroupKFold` does not fix it: over five
folds the closed share runs from 0.332 to 0.807, and the folds themselves come
out between 4 and 11 people, 6,253 and 26,372 frames. The notebook measures
both.

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
at `3e-4`, batch 256, 5 epochs — a few minutes on a consumer GPU.
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

## Running in detail

**Train.** `notebook.ipynb` top to bottom: it downloads the dataset through
`kagglehub` (no Kaggle token needed — the dataset is public), builds the
manifest, trains for five epochs and writes `best.pt`. A few minutes on a
consumer GPU, longer on a CPU.

**Experiments.** `experiments.ipynb` holds the comparisons behind every choice.
It reuses `data/manifest.csv` written by the first notebook, so run that one
first.

**Demo.** `python demo.py`. Quit with `q`, `Esc`, or by closing the window.

**Tests.** `pytest -q`. Ten tests drive `ClosureTracker` with a
fake clock: a blink of 0.15 s, a closure of 1.5 s that must stay silent, one of
2.5 s that must not, a face disappearing mid-closure. None of that is
reproducible in front of a camera.

**Two requirement files.** `requirements-lock.txt` pins the exact versions the
published numbers were measured with — use it if anything fails to import.
`requirements.txt` lists the same packages without versions, for a fresh
install on a newer Python.

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

The dataset, the manifest and the weights are not committed: all three are
reproducible from the code, and the weights are attached to the v1.0 release
so the demo can be run without training.

## Limitations

- Eight test subjects show a six-point spread in accuracy and are too few to
  bound it.
- Darkness was simulated, not filmed. Night and in-car conditions are untested.
- Some of the remaining errors belong to the dataset: among the most confident
  mistakes are runs of consecutive frames labelled "open" on which every model
  trained here answers "closed".
