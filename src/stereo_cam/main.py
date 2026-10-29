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
    FaceDetector,
    FaceDetectorOptions,
    GestureRecognizer,
    GestureRecognizerOptions,
    RunningMode,
)

from .animation import Spring, WordTracker, fade_out, pop
from .config import Config, ConfigError, load_config
from .drawing import (
    CAPTION_CAP_HEIGHT,
    HUD,
    caption_colour,
    draw_caption,
    draw_centered_text,
    draw_hand,
    hex_to_rgb,
)
from .effects import EFFECTS
from .fonts import FontRotation
from .frame_reader import CameraError, FrameReader
from .gestures import hand_gestures, is_trigger_sign
from .hold_timer import HoldTimer
from .speech import Transcriber

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "gesture_recognizer.task"
FACE_MODEL_PATH = Path(__file__).parent / "models" / "blaze_face_short_range.tflite"
FACE_SMOOTHING = 0.7  # closer to 1 = steadier face box but slower to follow
FOLLOW_FREQUENCY = 7.0  # how quickly the caption follows your head; higher is snappier
FOLLOW_DAMPING = 0.6  # 1.0 settles with no bounce; lower overshoots a little more
EMPHASIS_SWELL = (
    0.8  # how much bigger a word at full emphasis is drawn, at caption_emphasis 1.0
)
EMPHASIS_STEP = 0.05  # sizes are rounded to this, so only so many font sizes are made
TILT_FREQUENCY = 6.0  # how quickly the caption turns with your head
TILT_DAMPING = 0.8  # 1.0 settles with no overshoot
TILT_DEAD_ZONE = math.radians(
    3
)  # head tilts smaller than this are mostly detector noise
TILT_MAX = math.radians(35)  # the caption never turns further than this
FACE_LOST_SECONDS = 1.0  # keep the last face box this long when detection drops out
REFERENCE_FACE_WIDTH = (
    220  # face width in pixels at which captions are their default size
)
CAPTION_SCALE_RANGE = (0.5, 2.5)  # how far the face can change the caption size
FALLBACK_FPS = 30  # used when the camera does not report its frame rate
MAX_HANDS = 2  # hands the recognizer tracks, and the gesture lines the HUD shows

# Keys
KEY_CAPTURE = ord("c")
KEY_RESET = ord("r")
KEY_LANGUAGE = ord("l")
KEY_QUIT = ord("q")

FPS_SMOOTHING = 0.9  # closer to 1 = steadier but slower to react


def smooth(smoothed, value):
    """Blend a new measurement into a running average (0.0 means no average yet)."""
    if smoothed == 0.0:
        return value

    return FPS_SMOOTHING * smoothed + (1 - FPS_SMOOTHING) * value


def head_roll(keypoints, width, height):
    """How far the head is tilted, in radians, clockwise on screen, from the
    line through the eyes and the line through the ears. 0.0 if the detector
    didn't report them."""
    if len(keypoints) < 6:
        return 0.0

    # the detector lists: eye, eye, nose, mouth, ear, ear
    across_x = across_y = 0.0
    for pair in (keypoints[0:2], keypoints[4:6]):
        (left_x, left_y), (right_x, right_y) = sorted(
            (point.x * width, point.y * height) for point in pair
        )
        across_x += right_x - left_x
        across_y += right_y - left_y

    return math.atan2(across_y, across_x)


def detect_face(face_detector, frame, now):
    """The most confident face in the frame as (x, y, width, height, roll),
    or None."""
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detections = face_detector.detect_for_video(mp_image, int(now * 1000)).detections
    if not detections:
        return None

    best = max(detections, key=lambda d: d.categories[0].score)
    box = best.bounding_box
    roll = head_roll(best.keypoints, frame.shape[1], frame.shape[0])
    return (box.origin_x, box.origin_y, box.width, box.height, roll)


def tilt_target(roll):
    """The angle the caption should turn to for a head roll: small tilts are
    ignored (the key points are noisy), and big ones are limited."""
    magnitude = max(abs(roll) - TILT_DEAD_ZONE, 0.0)
    return math.copysign(min(magnitude, TILT_MAX), roll)


def caption_scale(face, setting):
    """How big to draw the caption: in proportion to the face, so it looks the
    same whether you sit close or far. Rounded to steps of 5%, which keeps the
    number of different font sizes down."""
    scale = setting
    if face is not None:
        low, high = CAPTION_SCALE_RANGE
        scale *= min(max(face[2] / REFERENCE_FACE_WIDTH, low), high)

    return round(scale * 20) / 20


def smooth_box(smoothed, box):
    """Blend a new face box into the running one (None means no box yet)."""
    if smoothed is None:
        return box

    return tuple(
        FACE_SMOOTHING * old + (1 - FACE_SMOOTHING) * new
        for old, new in zip(smoothed, box, strict=True)
    )


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


def run(
    frame_reader,
    cam,
    recognizer,
    background,
    config,
    transcriber=None,
    face_detector=None,
):
    hud = HUD(MAX_HANDS)
    log.info("Using virtual camera: %s", cam.device)
    last = time.perf_counter()

    state = State.LIVE
    dissolve_timer = config.dissolve_duration
    dissolve_direction = 1

    hold_timer = HoldTimer(config.hold_duration, config.hold_grace)
    effect = EFFECTS[config.effect]
    caption_fonts = FontRotation(config.caption_font_seconds)
    caption_words = WordTracker()
    highlight = hex_to_rgb(config.caption_highlight)
    caption_follow = Spring(FOLLOW_FREQUENCY, FOLLOW_DAMPING)
    caption_tilt = Spring(TILT_FREQUENCY, TILT_DAMPING)
    face_box = None  # smoothed (x, y, width, height, roll) of your face, if tracking
    face_seen_at = 0.0
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

        if face_detector is not None:
            detected = detect_face(face_detector, frame, now)
            if detected is not None:
                face_box = smooth_box(face_box, detected)
                face_seen_at = now
            elif now - face_seen_at > FACE_LOST_SECONDS:
                face_box = None

        # runs every frame, so a caption that appears later starts at your head
        caption_face = None
        tilt = 0.0
        if face_box is None:
            caption_follow.reset()
            caption_tilt.reset()
        else:
            x, y = caption_follow.update(face_box[:2], dt)
            caption_face = [round(x), round(y), round(face_box[2]), round(face_box[3])]
            tilt = caption_tilt.update((tilt_target(face_box[4]),), dt)[0]

        output = effect(frame, background, 1 - live_weight)

        # Captions belong to you: they fade out as you dissolve and back in as
        # you return, and stay away while you are out of the frame.
        presence = live_weight
        if face_detector is not None and face_box is None:
            presence = 0.0

        if transcriber is not None:
            heard, heard_at, phrase = transcriber.latest()
            showing = bool(heard) and now - heard_at < config.caption_seconds
            shown_words = caption_words.update(heard if showing else [], phrase, now)
            if showing and presence > 0:
                words = []
                for word, age, emphasis in shown_words:
                    # squared, so a word that is only a bit louder stays close to normal size
                    swell = 1 + EMPHASIS_SWELL * emphasis**2 * config.caption_emphasis
                    pop_scale, alpha = pop(age)
                    swell = round(swell / EMPHASIS_STEP) * EMPHASIS_STEP
                    shade = min(emphasis * config.caption_emphasis, 1.0)
                    colour = caption_colour(highlight, shade)
                    words.append((word, swell, pop_scale, alpha, colour))
                scale = caption_scale(caption_face, config.caption_scale)
                text = " ".join(word[0] for word in words)
                font = caption_fonts.current(
                    now, round(CAPTION_CAP_HEIGHT * scale), text
                )
                fade = fade_out(now, heard_at, config.caption_seconds) * presence
                draw_caption(
                    output,
                    words,
                    font,
                    caption_face,
                    scale,
                    fade,
                    tilt * config.caption_tilt,
                )

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
        if transcriber is not None:
            label = f"Captions: {transcriber.language.upper()}  (L to switch)"
            cv.putText(
                debug_output,
                label,
                (10, debug_output.shape[0] - 15),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
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
        elif key == KEY_LANGUAGE and transcriber is not None:
            languages = config.speech_languages
            following = languages[
                (languages.index(transcriber.language) + 1) % len(languages)
            ]
            transcriber.set_language(following)
            log.info("Captions language: %s", following)
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
    if config.track_face and not FACE_MODEL_PATH.exists():
        log.error("Face model not found: %s", FACE_MODEL_PATH)
        log.error("Download it with the curl command listed in .gitignore")
        sys.exit(1)

    cap = open_camera(config.camera_index, config.requested_fps)
    frame_reader = FrameReader(cap)

    try:
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS) or FALLBACK_FPS

        log.info("Camera Resolution: %s x %s", width, height)
        log.info("Camera FPS: %s", fps)

        background = capture_background(frame_reader, config.countdown_seconds)

        transcriber = None
        if config.captions:
            transcriber = Transcriber(
                config.speech_model,
                config.speech_languages[0],
                None if config.microphone < 0 else config.microphone,
            )
            transcriber.start()

        face_detector = None
        if config.track_face:
            face_detector = FaceDetector.create_from_options(
                FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
                    running_mode=RunningMode.VIDEO,
                )
            )

        try:
            with GestureRecognizer.create_from_options(options) as recognizer:  # noqa: SIM117
                with pvm.Camera(width=width, height=height, fps=fps) as cam:
                    run(
                        frame_reader,
                        cam,
                        recognizer,
                        background,
                        config,
                        transcriber,
                        face_detector,
                    )
        finally:
            if face_detector is not None:
                face_detector.close()
            if transcriber is not None:
                transcriber.stop()

        log.info("# DONE #")
    except CameraError as error:
        log.error("%s. Exiting ...", error)
        sys.exit(1)
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
