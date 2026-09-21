"""Tests for the closure logic.

ClosureTracker takes a timestamp on every call, so the whole thing can be
driven by a fake clock: no camera, no model, no waiting. Holding your eyes shut
for exactly 1.9 seconds in front of a webcam is not a repeatable test — this is.

Run:  pytest -q
"""
import pytest

from tracker import ClosureTracker

FPS = 30
STEP = 1 / FPS
OPEN, CLOSED = 0.02, 1.0     # what the classifier returns in practice


def feed(tracker, prob, seconds, now):
    """Feed frames at 30 fps. Returns the last state and the new time."""
    state = tracker.state(now)
    for _ in range(round(seconds / STEP)):
        now += STEP
        state = tracker.update(prob, now)
    return state, now


def lose_face(tracker, seconds, now):
    """Same, but the face is not found in any of the frames."""
    state = tracker.state(now)
    for _ in range(round(seconds / STEP)):
        now += STEP
        state = tracker.miss(now)
    return state, now


@pytest.fixture
def tracker():
    return ClosureTracker(threshold=0.5, alarm_after=2.0, window=60.0)


def test_open_eyes_never_alarm(tracker):
    state, _ = feed(tracker, OPEN, 10.0, now=0.0)
    assert not state["closed"]
    assert state["duration"] == 0.0
    assert not state["alarm"]


def test_blink_does_not_alarm(tracker):
    _, now = feed(tracker, OPEN, 3.0, now=0.0)
    state, now = feed(tracker, CLOSED, 0.15, now)      # a blink
    assert state["duration"] < 0.5
    assert not state["alarm"]

    state, _ = feed(tracker, OPEN, 1.0, now)
    assert state["duration"] == 0.0                     # counter reset


def test_alarm_fires_only_past_the_threshold(tracker):
    _, now = feed(tracker, OPEN, 1.0, now=0.0)

    state, now = feed(tracker, CLOSED, 1.5, now)
    assert state["closed"] and not state["alarm"]       # too early

    state, _ = feed(tracker, CLOSED, 1.0, now)
    assert state["alarm"]
    assert state["duration"] >= 2.0


def test_squint_is_not_a_closure(tracker):
    """The two eyes disagreeing averages to something mid-range. That is a
    genuinely ambiguous frame and must not start the counter."""
    state, _ = feed(tracker, 0.42, 3.0, now=0.0)
    assert not state["closed"]
    assert not state["alarm"]


def test_single_noisy_frame_does_not_flip_the_state(tracker):
    """Smoothing exists so one bad frame cannot change the verdict."""
    _, now = feed(tracker, OPEN, 2.0, now=0.0)
    state = tracker.update(CLOSED, now + STEP)          # exactly one frame
    assert not state["closed"]


def test_opening_the_eyes_resets_the_counter(tracker):
    _, now = feed(tracker, CLOSED, 3.0, now=0.0)
    state, now = feed(tracker, OPEN, 1.0, now)
    assert state["duration"] == 0.0
    assert not state["alarm"]


def test_lost_face_keeps_the_counter_running(tracker):
    """A sleeping driver's head drops and the face leaves the frame exactly
    when the alarm matters most, so losing it must not reset the timer."""
    _, now = feed(tracker, OPEN, 1.0, now=0.0)
    state, now = feed(tracker, CLOSED, 1.2, now)
    assert not state["alarm"]

    state, _ = lose_face(tracker, 1.5, now)
    assert state["alarm"]
    assert state["lost"] > 1.0


def test_lost_face_while_awake_does_not_alarm(tracker):
    _, now = feed(tracker, OPEN, 2.0, now=0.0)
    state, _ = lose_face(tracker, 5.0, now)
    assert not state["alarm"]
    assert state["duration"] == 0.0


def test_face_coming_back_clears_the_lost_flag(tracker):
    _, now = feed(tracker, OPEN, 1.0, now=0.0)
    _, now = lose_face(tracker, 3.0, now)
    state, _ = feed(tracker, OPEN, 0.5, now)
    assert state["lost"] == 0.0


def test_perclos_measures_the_share_of_closed_time(tracker):
    _, now = feed(tracker, OPEN, 30.0, now=0.0)
    assert tracker.perclos() == pytest.approx(0.0, abs=0.01)

    _, _ = feed(tracker, CLOSED, 30.0, now)
    assert tracker.perclos() == pytest.approx(0.5, abs=0.02)


def test_perclos_forgets_beyond_its_window():
    """Only the last minute counts, otherwise the measure would drift for the
    whole session instead of describing the driver's current state."""
    tracker = ClosureTracker(window=10.0)
    _, now = feed(tracker, CLOSED, 10.0, now=0.0)
    assert tracker.perclos() == pytest.approx(1.0, abs=0.02)

    _, _ = feed(tracker, OPEN, 10.0, now)
    assert tracker.perclos() == pytest.approx(0.0, abs=0.02)


def test_alarm_threshold_is_configurable():
    tracker = ClosureTracker(alarm_after=5.0)
    state, _ = feed(tracker, CLOSED, 3.0, now=0.0)
    assert state["closed"] and not state["alarm"]
