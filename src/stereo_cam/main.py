import argparse
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

from .config import Config, ConfigError, load_config
from .drawing import HUD, draw_centered_text, draw_hand
from .effects import EFFECTS
from .frame_reader import CameraError, FrameReader
from .gestures import hand_gestures, is_trigger_sign
from .hold_timer import HoldTimer

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "gesture_recognizer.task"
FALLBACK_FPS = 30  # used when the camera does not report its frame rate
MAX_HANDS = 2  # hands the recognizer tracks, and the gesture lines the HUD shows

# Keys
KEY_CAPTURE = ord("c")
KEY_RESET = ord("r")
KEY_QUIT = ord("q")

FPS_SMOOTHING = 0.9  # closer to 1 = steadier but slower to react


def smooth(smoothed, value):
    """Blend a new measurement into a running average (0.0 means no average yet)."""
    if smoothed == 0.0:
        return value

    return FPS_SMOOTHING * smoothed + (1 - FPS_SMOOTHING) * value


class State(Enum):
    LIVE = auto()
    DISSOLVING = auto()
    GONE = auto()


def open_camera(index, requested_fps):
    cap = cv.VideoCapture(index)
    if not cap.isOpened():
        log.error("Cannot open camera")
        sys.exit(1)

    cap.set(cv.CAP_PROP_FPS, requested_fps)

    return cap


def capture_background(frame_reader, countdown_seconds):
    started = None

    while True:
        frame = frame_reader.read()
        preview = frame.copy()

        if started is not None:
            now = time.perf_counter()
            elapsed = now - started

            remaining = math.ceil(countdown_seconds - elapsed)
            draw_centered_text(preview, str(remaining), 12, 5)

            if elapsed > countdown_seconds:
                cv.destroyWindow("background")
                return frame
        else:
            draw_centered_text(preview, "Press 'c' to start the countdown", 3, 1)

        cv.imshow("background", preview)

        key = cv.waitKey(1) & 0xFF
        if key == KEY_CAPTURE and started is None:
            started = time.perf_counter()


def run(frame_reader, cam, recognizer, background, config):
    hud = HUD(MAX_HANDS)
    log.info("Using virtual camera: %s", cam.device)
    last = time.perf_counter()

    state = State.LIVE
    dissolve_timer = config.dissolve_duration
    dissolve_direction = 1

    hold_timer = HoldTimer(config.hold_duration, config.hold_grace)
    effect = EFFECTS[config.effect]
    fps_display = 0.0
    latency_display = 0.0  # ms of our own processing per frame

    while True:
        now = time.perf_counter()
        dt = now - last
        last = now

        if dt > 0:
            fps_display = smooth(fps_display, 1 / dt)

        frame = frame_reader.read()
        frame_read_at = time.perf_counter()

        if state == State.DISSOLVING:
            dissolve_timer = min(
                config.dissolve_duration,
                max(0, dissolve_timer - dt * dissolve_direction),
            )
            if dissolve_direction == 1 and dissolve_timer == 0:
                state = State.GONE
                dissolve_direction = -1
            elif (
                dissolve_direction == -1 and dissolve_timer == config.dissolve_duration
            ):
                state = State.LIVE
                dissolve_direction = 1

        live_weight = dissolve_timer / config.dissolve_duration

        output = effect(frame, background, 1 - live_weight)

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

            is_sign = is_trigger_sign(
                gestures, config.trigger_gesture, config.trigger_threshold
            )
            hold_progress = hold_timer.update(is_sign, now)
            if hold_progress >= 1.0:
                state = State.DISSOLVING
                hold_timer.reset()

            draw_hand(debug_output, landmarks)

        hud.draw(
            debug_output, fps_display, latency_display, state, gestures, hold_progress
        )
        cv.imshow("frame", debug_output)

        cam.send(cv.cvtColor(output, cv.COLOR_BGR2RGB))
        # Measured before the deliberate wait, so it isn't counted as lag. It
        # reaches the HUD one frame late, which is fine for a smoothed number.
        processing_ms = (time.perf_counter() - frame_read_at) * 1000
        latency_display = smooth(latency_display, processing_ms)
        cam.sleep_until_next_frame()

        key = cv.waitKey(1) & 0xFF
        if key == KEY_RESET:
            state = State.LIVE
            dissolve_timer = config.dissolve_duration
            dissolve_direction = 1
            hold_timer.reset()
        elif key == KEY_QUIT:
            break


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log.info("# STEREO CAM #")

    parser = argparse.ArgumentParser(description="Dissolve your webcam on a hand sign.")
    parser.add_argument("--config", help="path to a TOML settings file")
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else Config()
    except ConfigError as error:
        log.error("%s", error)
        sys.exit(1)
    log.info("Config: %s", config)

    if not MODEL_PATH.exists():
        log.error("Model not found: %s", MODEL_PATH)
        log.error("Download it with the curl command listed in .gitignore")
        sys.exit(1)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=MAX_HANDS,
    )
    cap = open_camera(config.camera_index, config.requested_fps)
    frame_reader = FrameReader(cap)

    try:
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS) or FALLBACK_FPS

        log.info("Camera Resolution: %s x %s", width, height)
        log.info("Camera FPS: %s", fps)

        background = capture_background(frame_reader, config.countdown_seconds)

        with GestureRecognizer.create_from_options(options) as recognizer:  # noqa: SIM117
            with pvm.Camera(width=width, height=height, fps=fps) as cam:
                run(frame_reader, cam, recognizer, background, config)

        log.info("# DONE #")
    except CameraError as error:
        log.error("%s. Exiting ...", error)
        sys.exit(1)
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
