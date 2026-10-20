import functools
import logging
import math
import sys
import time
from enum import Enum, auto
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
import pyvirtualcam as pvm
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    HandLandmarksConnections,
    RunningMode,
)

from .frame_reader import CameraError, FrameReader
from .hold_timer import HoldTimer

log = logging.getLogger(__name__)

DISSOLVE_DURATION = 1  # seconds
MODEL_PATH = Path(__file__).parent / "models" / "gesture_recognizer.task"
TRIGGER_GESTURE = "Victory"
TRIGGER_THRESHOLD = 0.8  # confidence threshold for the trigger gesture
FALLBACK_FPS = 30  # used when the camera does not report its frame rate
REQUESTED_FPS = 30  # frame rate we ask the camera for (it may refuse)
COUNTDOWN_SECONDS = 3
MAX_HANDS = 2  # hands the recognizer tracks, and the gesture lines the HUD shows

# Keys
KEY_CAPTURE = ord("c")
KEY_RESET = ord("r")
KEY_QUIT = ord("q")

# Colours (BGR)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Preview drawing (pixels)
FONT = cv.FONT_HERSHEY_SIMPLEX
LANDMARK_RADIUS = 8
PANEL_POS = (10, 10)  # margin from the top-left corner
PANEL_WIDTH = 320
# The panel's height is these rows added up: FPS, then (only while live) the
# gesture rows and hold bar, then the state label at the bottom.
FPS_ROW_HEIGHT = 54
LIVE_ROWS_HEIGHT = 121
STATE_ROW_HEIGHT = 53
PANEL_RADIUS = 14  # corner roundness
PANEL_COLOR = (20, 20, 20)
PANEL_ALPHA = 0.6  # 0 = invisible, 1 = solid
SEPARATOR_COLOR = (200, 200, 200)
SEPARATOR_MARGIN = 10  # how far the separator lines stop short of the panel edges
# Positions inside the panel, relative to its top-left corner
FPS_TEXT_POS = (10, 30)
GESTURE_TEXT_POS = (10, 92)
GESTURE_LINE_HEIGHT = 35  # gap between the lines of two hands
BAR_X, BAR_Y, BAR_H = 10, 145, 14
BAR_W = PANEL_WIDTH - 2 * BAR_X  # full width, with the same margin on both sides
STATE_LABEL_X = 10
STATE_LABEL_BOTTOM_MARGIN = 15  # from the bottom edge of the panel
FPS_SMOOTHING = 0.9  # closer to 1 = steadier but slower to react


class State(Enum):
    LIVE = auto()
    DISSOLVING = auto()
    GONE = auto()


def open_camera():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open camera")
        sys.exit(1)

    cap.set(cv.CAP_PROP_FPS, REQUESTED_FPS)

    return cap


def draw_centered_text(img, text, scale, thickness):
    (text_width, text_height), _ = cv.getTextSize(text, FONT, scale, thickness)
    x = (img.shape[1] - text_width) // 2
    y = (img.shape[0] + text_height) // 2
    cv.putText(img, text, (x, y), FONT, scale, WHITE, thickness)


def capture_background(frame_reader):
    started = None

    while True:
        frame = frame_reader.read()
        preview = frame.copy()

        if started is not None:
            now = time.perf_counter()
            elapsed = now - started

            remaining = math.ceil(COUNTDOWN_SECONDS - elapsed)
            draw_centered_text(preview, str(remaining), 12, 5)

            if elapsed > COUNTDOWN_SECONDS:
                cv.destroyWindow("background")
                return frame
        else:
            draw_centered_text(preview, "Press 'c' to start the countdown", 3, 1)

        cv.imshow("background", preview)

        key = cv.waitKey(1) & 0xFF
        if key == KEY_CAPTURE and started is None:
            started = time.perf_counter()


def draw_hand(img, landmarks):
    if not landmarks:
        return

    img_width = img.shape[1]
    img_height = img.shape[0]

    for landmark in landmarks:
        points = [(int(p.x * img_width), int(p.y * img_height)) for p in landmark]
        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            start = points[connection.start]
            end = points[connection.end]
            cv.line(img, start, end, WHITE, 2)

        for point in points:
            cv.circle(img, point, LANDMARK_RADIUS, GREEN, -1)


def hand_gestures(result):
    """The top gesture of every detected hand, as a list of name and score."""
    return [
        {"name": hand[0].category_name, "score": hand[0].score}
        for hand in result.gestures
        if hand
    ]


def is_trigger_sign(gestures):
    """True when either hand makes the trigger sign with enough confidence."""
    return any(
        gesture["name"] == TRIGGER_GESTURE and gesture["score"] >= TRIGGER_THRESHOLD
        for gesture in gestures
    )


@functools.cache
def panel_alpha(width, height, radius):
    """How opaque the panel is at every pixel: rounded, with soft edges."""
    mask = np.zeros((height, width), np.uint8)
    cv.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
    cv.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
    for corner_x in (radius, width - radius - 1):
        for corner_y in (radius, height - radius - 1):
            cv.circle(mask, (corner_x, corner_y), radius, 255, -1, cv.LINE_AA)

    return (mask / 255.0 * PANEL_ALPHA).astype(np.float32)[:, :, None]


def draw_panel(img, height):
    """Darken a rounded rectangle of the image, so text on it stays readable."""
    x, y = PANEL_POS
    area = img[y : y + height, x : x + PANEL_WIDTH]
    # an image smaller than the panel just gets a clipped panel
    alpha = panel_alpha(PANEL_WIDTH, height, PANEL_RADIUS)
    alpha = alpha[: area.shape[0], : area.shape[1]]
    area[:] = area * (1 - alpha) + np.array(PANEL_COLOR, np.float32) * alpha


def draw_hud(img, fps, state, gestures, hold_progress):
    """Draw the HUD: FPS, then while live (gestures is not None) the gesture
    rows and hold bar, then the state label as the last line. A separator
    line sits under the FPS and above the state."""
    live = gestures is not None
    height = FPS_ROW_HEIGHT + STATE_ROW_HEIGHT
    if live:
        height += LIVE_ROWS_HEIGHT
    draw_panel(img, height)
    panel_x, panel_y = PANEL_POS

    ## FPS
    fps_pos = (panel_x + FPS_TEXT_POS[0], panel_y + FPS_TEXT_POS[1])
    cv.putText(img, f"{fps:.0f} FPS", fps_pos, FONT, 1, WHITE, 2)

    ## Separators (when not live they are the same line)
    line_left = panel_x + SEPARATOR_MARGIN
    line_right = panel_x + PANEL_WIDTH - SEPARATOR_MARGIN
    for offset in (FPS_ROW_HEIGHT, height - STATE_ROW_HEIGHT):
        line_y = panel_y + offset
        cv.line(img, (line_left, line_y), (line_right, line_y), SEPARATOR_COLOR, 1)

    if live:
        ## Gesture Text (one line per possible hand)
        text_x = panel_x + GESTURE_TEXT_POS[0]
        text_y = panel_y + GESTURE_TEXT_POS[1]
        for i in range(MAX_HANDS):
            if i < len(gestures):
                line = f"{gestures[i]['name']}: {gestures[i]['score']:.2f}"
            else:
                line = "Hand not showing"
            cv.putText(
                img,
                line,
                (text_x, text_y + i * GESTURE_LINE_HEIGHT),
                FONT,
                1,
                WHITE,
                2,
            )

        ## Gesture Bar
        bar_x = panel_x + BAR_X
        bar_y = panel_y + BAR_Y
        cv.rectangle(img, (bar_x, bar_y), (bar_x + BAR_W, bar_y + BAR_H), WHITE, 1)
        cv.rectangle(
            img,
            (bar_x, bar_y),
            (bar_x + int(BAR_W * hold_progress), bar_y + BAR_H),
            GREEN,
            -1,
        )

    ## State (always the last line)
    state_pos = (
        panel_x + STATE_LABEL_X,
        panel_y + height - STATE_LABEL_BOTTOM_MARGIN,
    )
    cv.putText(img, state.name, state_pos, FONT, 1, WHITE, 2)


def run(frame_reader, cam, recognizer, background):
    log.info("Using virtual camera: %s", cam.device)
    last = time.perf_counter()

    state = State.LIVE
    dissolve_timer = DISSOLVE_DURATION
    dissolve_direction = 1

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

        frame = frame_reader.read()

        if state == State.DISSOLVING:
            dissolve_timer = min(
                DISSOLVE_DURATION,
                max(0, dissolve_timer - dt * dissolve_direction),
            )
            if dissolve_direction == 1 and dissolve_timer == 0:
                state = State.GONE
                dissolve_direction = -1
            elif dissolve_direction == -1 and dissolve_timer == DISSOLVE_DURATION:
                state = State.LIVE
                dissolve_direction = 1

        live_weight = dissolve_timer / DISSOLVE_DURATION
        output = cv.addWeighted(frame, live_weight, background, 1 - live_weight, 0)
        debug_output = output.copy()

        gestures = None
        hold_progress = 0.0
        if state in (State.LIVE, State.GONE):
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            gesture_recognition_result = recognizer.recognize_for_video(
                mp_image, int(now * 1000)
            )

            landmarks = gesture_recognition_result.hand_landmarks
            gestures = hand_gestures(gesture_recognition_result)

            is_sign = is_trigger_sign(gestures)
            hold_progress = hold_timer.update(is_sign, now)
            if hold_progress >= 1.0:
                state = State.DISSOLVING
                hold_timer.reset()

            draw_hand(debug_output, landmarks)

        draw_hud(debug_output, fps_display, state, gestures, hold_progress)
        cv.imshow("frame", debug_output)

        cam.send(cv.cvtColor(output, cv.COLOR_BGR2RGB))
        cam.sleep_until_next_frame()

        key = cv.waitKey(1) & 0xFF
        if key == KEY_RESET:
            state = State.LIVE
            dissolve_timer = DISSOLVE_DURATION
            dissolve_direction = 1
            hold_timer.reset()
        elif key == KEY_QUIT:
            break


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log.info("# EXIT CAM #")

    if not MODEL_PATH.exists():
        log.error("Model not found: %s", MODEL_PATH)
        log.error("Download it with the curl command listed in .gitignore")
        sys.exit(1)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=MAX_HANDS,
    )
    cap = open_camera()
    frame_reader = FrameReader(cap)

    try:
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS) or FALLBACK_FPS

        log.info("Camera Resolution: %s x %s", width, height)
        log.info("Camera FPS: %s", fps)

        background = capture_background(frame_reader)

        with GestureRecognizer.create_from_options(options) as recognizer:  # noqa: SIM117
            with pvm.Camera(width=width, height=height, fps=fps) as cam:
                run(frame_reader, cam, recognizer, background)

        log.info("# DONE #")
    except CameraError as error:
        log.error("%s. Exiting ...", error)
        sys.exit(1)
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
