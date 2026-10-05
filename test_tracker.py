import pytest
from tracker import ClosureTracker

FPS = 30
STEP = 1 / FPS
OPEN, CLOSED = 0.02, 1.0


def feed(tracker, prob, seconds, now):
    state = tracker.state(now)
    for _ in range(round(seconds / STEP)):
        now += STEP
        state = tracker.update(prob, now)
    return state, now


def lose_face(tracker, seconds, now):
    state = tracker.state(now)
    for _ in range(round(seconds / STEP)):
        now += STEP
        state = tracker.miss(now)
    return state, now


@pytest.fixture
def tracker():
    return ClosureTracker(threshold=0.5, alarm_after=2.0)


def test_open_eyes_never_alarm(tracker):
    state, _ = feed(tracker, OPEN, 10.0, now=0.0)
    assert not state["closed"]
    assert state["duration"] == 0.0
    assert not state["alarm"]


def test_blink_does_not_alarm(tracker):
    _, now = feed(tracker, OPEN, 3.0, now=0.0)
    state, now = feed(tracker, CLOSED, 0.15, now)
    assert state["duration"] < 0.5
    assert not state["alarm"]

    state, _ = feed(tracker, OPEN, 1.0, now)
    assert state["duration"] == 0.0


def test_alarm_fires_only_past_the_threshold(tracker):
    _, now = feed(tracker, OPEN, 1.0, now=0.0)

    state, now = feed(tracker, CLOSED, 1.5, now)
    assert state["closed"] and not state["alarm"]

    state, _ = feed(tracker, CLOSED, 1.0, now)
    assert state["alarm"]
    assert state["duration"] >= 2.0


def test_squint_is_not_a_closure(tracker):
    state, _ = feed(tracker, 0.42, 3.0, now=0.0)
    assert not state["closed"]
    assert not state["alarm"]


def test_single_noisy_frame_does_not_flip_the_state(tracker):
    _, now = feed(tracker, OPEN, 2.0, now=0.0)
    state = tracker.update(CLOSED, now + STEP)
    assert not state["closed"]


def test_opening_the_eyes_resets_the_counter(tracker):
    _, now = feed(tracker, CLOSED, 3.0, now=0.0)
    state, now = feed(tracker, OPEN, 1.0, now)
    assert state["duration"] == 0.0
    assert not state["alarm"]


def test_lost_face_keeps_the_counter_running(tracker):
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


def test_alarm_threshold_is_configurable():
    tracker = ClosureTracker(alarm_after=5.0)
    state, _ = feed(tracker, CLOSED, 3.0, now=0.0)
    assert state["closed"] and not state["alarm"]
