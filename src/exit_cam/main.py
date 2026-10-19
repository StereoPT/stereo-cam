import sys
import time
from enum import Enum, auto
from pathlib import Path
import cv2 as cv
import mediapipe as mp
import pyvirtualcam as pvm

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    GestureRecognizerOptions,
    GestureRecognizer,
    HandLandmarksConnections,
    RunningMode,
)

print("# EXIT CAM #")

DISSOLVE_DURATION = 1  # seconds
MODEL_PATH = Path(__file__).parent / "models" / "gesture_recognizer.task"
TRIGGER_GESTURE = "Victory"
HOLD_DURATION = 0.5  # seconds the sign must be held
HOLD_GRACE = 0.1  # seconds the sign may drop out without resetting the hold
FALLBACK_FPS = 30  # used when the camera does not report its frame rate
REQUESTED_FPS = 30  # frame rate we ask the camera for (it may refuse)

# Keys
KEY_CAPTURE = ord("c")
KEY_RESET = ord("r")
KEY_QUIT = ord("q")

# Colours (BGR)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Preview drawing (pixels)
FONT = cv.FONT_HERSHEY_SIMPLEX
LANDMARK_RADIUS = 8
INDICATOR_POS = (25, 30)
INDICATOR_RADIUS = 12
GESTURE_TEXT_POS = (50, 40)
BAR_X, BAR_Y, BAR_W, BAR_H = 10, 60, 200, 14
STATE_LABEL_X = 10
STATE_LABEL_BOTTOM_MARGIN = 15
FPS_RIGHT_MARGIN = 140
FPS_SMOOTHING = 0.9  # closer to 1 = steadier but slower to react


class State(Enum):
    LIVE = auto()
    DISSOLVING = auto()
    GONE = auto()


class HoldTimer:
    """Tracks how long a sign has been held, forgiving brief dropouts."""

    def __init__(self, duration=HOLD_DURATION, grace=HOLD_GRACE):
        self.duration = duration
        self.grace = grace
        self.reset()

    def reset(self):
        self.held_since = None
        self.last_seen = None

    def update(self, is_held, now):
        """Feed one frame's result. Returns the hold progress from 0.0 to 1.0."""
        if is_held:
            if self.held_since is None:
                self.held_since = now
            self.last_seen = now
        elif self.held_since is not None and now - self.last_seen > self.grace:
            self.reset()

        if self.held_since is None:
            return 0.0

        return min(1.0, (now - self.held_since) / self.duration)


def open_camera():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit(1)

    cap.set(cv.CAP_PROP_FPS, REQUESTED_FPS)

    return cap


def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        sys.exit(1)

    return frame


def capture_background(cap):
    print("Press 'c' to capture a frame:")

    while True:
        frame = read_frame(cap)

        cv.imshow("background", frame)

        key = cv.waitKey(1) & 0xFF
        if key == KEY_CAPTURE:
            cv.destroyWindow("background")
            return frame


def draw_hand(img, landmarks):
    if not landmarks:
        return

    img_width = img.shape[1]
    img_height = img.shape[0]

    points = [(int(p.x * img_width), int(p.y * img_height)) for p in landmarks[0]]
    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        start = points[connection.start]
        end = points[connection.end]
        cv.line(img, start, end, WHITE, 2)

    for point in points:
        cv.circle(img, point, LANDMARK_RADIUS, GREEN, -1)


def draw_state_label(img, state):
    cv.putText(
        img,
        state.name,
        (STATE_LABEL_X, img.shape[0] - STATE_LABEL_BOTTOM_MARGIN),
        FONT,
        1,
        WHITE,
        2,
    )


def draw_fps(img, fps):
    cv.putText(
        img,
        f"{fps:.0f} FPS",
        (img.shape[1] - FPS_RIGHT_MARGIN, 30),
        FONT,
        1,
        WHITE,
        2,
    )


def top_gesture(result):
    gesture_name, gesture_score = "none", 0.0
    if result.gestures:
        best = result.gestures[0][0]
        gesture_name, gesture_score = best.category_name, best.score

    return {"name": gesture_name, "score": gesture_score}


def draw_hud(img, is_sign, gesture_name, gesture_score, hold_progress):
    ## Indicator Circle
    indicator_color = GREEN if is_sign else RED
    cv.circle(img, INDICATOR_POS, INDICATOR_RADIUS, indicator_color, -1)

    ## Gesture Text
    cv.putText(
        img,
        f"{gesture_name} {gesture_score:.2f}",
        GESTURE_TEXT_POS,
        FONT,
        1,
        WHITE,
        2,
    )

    ## Gesture Bar
    cv.rectangle(
        img,
        (BAR_X, BAR_Y),
        (BAR_X + BAR_W, BAR_Y + BAR_H),
        WHITE,
        1,
    )
    cv.rectangle(
        img,
        (BAR_X, BAR_Y),
        (BAR_X + int(BAR_W * hold_progress), BAR_Y + BAR_H),
        GREEN,
        -1,
    )


def run(cap, cam, recognizer, background):
    print(f"Using virtual camera: {cam.device}")
    last = time.perf_counter()

    state = State.LIVE
    dissolve_timer = DISSOLVE_DURATION

    hold_timer = HoldTimer()
    fps_display = 0.0

    while True:
        now = time.perf_counter()
        dt = now - last
        last = now

        if dt > 0:
            instant_fps = 1 / dt
            fps_display = (
                instant_fps
                if fps_display == 0.0
                else FPS_SMOOTHING * fps_display + (1 - FPS_SMOOTHING) * instant_fps
            )

        frame = read_frame(cap)

        if state == State.DISSOLVING:
            dissolve_timer = max(0, dissolve_timer - dt)
            if dissolve_timer == 0:
                state = State.GONE

        live_weight = dissolve_timer / DISSOLVE_DURATION
        output = cv.addWeighted(frame, live_weight, background, 1 - live_weight, 0)
        debug_output = output.copy()

        # Detection only matters while we are still live; once the exit has
        # started there is nothing left to watch for.
        if state == State.LIVE:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            gesture_recognition_result = recognizer.recognize_for_video(
                mp_image, int(now * 1000)
            )

            landmarks = gesture_recognition_result.hand_landmarks
            top = top_gesture(gesture_recognition_result)

            is_sign = top["name"] == TRIGGER_GESTURE
            hold_progress = hold_timer.update(is_sign, now)
            if hold_progress >= 1.0:
                state = State.DISSOLVING
                hold_timer.reset()

            draw_hand(debug_output, landmarks)
            draw_hud(debug_output, is_sign, top["name"], top["score"], hold_progress)

        draw_state_label(debug_output, state)
        draw_fps(debug_output, fps_display)
        cv.imshow("frame", debug_output)

        cam.send(cv.cvtColor(output, cv.COLOR_BGR2RGB))
        cam.sleep_until_next_frame()

        key = cv.waitKey(1) & 0xFF
        if key == KEY_RESET:
            state = State.LIVE
            dissolve_timer = DISSOLVE_DURATION
            hold_timer.reset()
        elif key == KEY_QUIT:
            break


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Download it with the curl command listed in .gitignore")
        sys.exit(1)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
    )
    cap = open_camera()

    try:
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS) or FALLBACK_FPS

        print("Camera Resolution:", width, "x", height)
        print("Camera FPS:", fps)

        background = capture_background(cap)

        with GestureRecognizer.create_from_options(options) as recognizer:
            with pvm.Camera(width=width, height=height, fps=fps) as cam:
                run(cap, cam, recognizer, background)

        print("# DONE #")
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
