import logging
import math
import sys
import time
from enum import Enum, auto
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import pyvirtualcam as pvm
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    RunningMode,
)

from .drawing import HUD, draw_centered_text, draw_hand
from .frame_reader import CameraError, FrameReader
from .gestures import hand_gestures, is_trigger_sign
from .hold_timer import HoldTimer

log = logging.getLogger(__name__)

DISSOLVE_DURATION = 1  # seconds
MODEL_PATH = Path(__file__).parent / "models" / "gesture_recognizer.task"
FALLBACK_FPS = 30  # used when the camera does not report its frame rate
REQUESTED_FPS = 30  # frame rate we ask the camera for (it may refuse)
COUNTDOWN_SECONDS = 3
MAX_HANDS = 2  # hands the recognizer tracks, and the gesture lines the HUD shows

# Keys
KEY_CAPTURE = ord("c")
KEY_RESET = ord("r")
KEY_QUIT = ord("q")

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


def run(frame_reader, cam, recognizer, background):
    hud = HUD(MAX_HANDS)
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

        hud.draw_hud()
        hud.draw(debug_output, fps_display, state, gestures, hold_progress)
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
