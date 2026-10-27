import functools

import cv2 as cv
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

# Colours (BGR)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

FONT = cv.FONT_HERSHEY_SIMPLEX
LANDMARK_RADIUS = 8

PANEL_POS = (10, 10)  # margin from the top-left corner
PANEL_WIDTH = 320
PANEL_RADIUS = 14  # corner roundness
PANEL_COLOR = (20, 20, 20)
PANEL_ALPHA = 0.6  # 0 = invisible, 1 = solid
# The panel's height is these rows added up: FPS, then (only while live) the
# gesture rows and hold bar, then the state label at the bottom.
FPS_ROW_HEIGHT = 54
LIVE_ROWS_HEIGHT = 121
STATE_ROW_HEIGHT = 53
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


def draw_centered_text(img, text, scale, thickness):
    (text_width, text_height), _ = cv.getTextSize(text, FONT, scale, thickness)
    x = (img.shape[1] - text_width) // 2
    y = (img.shape[0] + text_height) // 2
    cv.putText(img, text, (x, y), FONT, scale, WHITE, thickness)


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


@functools.cache
def _panel_alpha(width, height, radius):
    """How opaque the panel is at every pixel: rounded, with soft edges."""
    mask = np.zeros((height, width), np.uint8)
    cv.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
    cv.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
    for corner_x in (radius, width - radius - 1):
        for corner_y in (radius, height - radius - 1):
            cv.circle(mask, (corner_x, corner_y), radius, 255, -1, cv.LINE_AA)

    return (mask / 255.0 * PANEL_ALPHA).astype(np.float32)[:, :, None]


class HUD:
    """The on-screen panel: FPS and processing time, then while live (gestures is not None) the
    gesture rows and hold bar, then the state label as the last line. A
    separator line sits under the FPS and above the state."""

    def __init__(self, max_hands):
        self.max_hands = max_hands  # gesture lines to show, one per possible hand

    def draw(self, img, fps, latency_ms, state, gestures, hold_progress):
        live = gestures is not None
        height = FPS_ROW_HEIGHT + STATE_ROW_HEIGHT
        if live:
            height += LIVE_ROWS_HEIGHT

        self._draw_panel(img, height)
        self._draw_fps(img, fps, latency_ms)
        self._draw_separators(img, height)
        if live:
            self._draw_gestures(img, gestures)
            self._draw_bar(img, hold_progress)
        self._draw_state(img, state, height)

    def _draw_panel(self, img, height):
        """Darken a rounded rectangle of the image, so text on it stays readable."""
        x, y = PANEL_POS
        area = img[y : y + height, x : x + PANEL_WIDTH]
        # an image smaller than the panel just gets a clipped panel
        alpha = _panel_alpha(PANEL_WIDTH, height, PANEL_RADIUS)
        alpha = alpha[: area.shape[0], : area.shape[1]]
        area[:] = area * (1 - alpha) + np.array(PANEL_COLOR, np.float32) * alpha

    def _draw_fps(self, img, fps, latency_ms):
        pos = (PANEL_POS[0] + FPS_TEXT_POS[0], PANEL_POS[1] + FPS_TEXT_POS[1])
        cv.putText(img, f"{fps:.0f} FPS  {latency_ms:.0f} ms", pos, FONT, 1, WHITE, 2)

    def _draw_separators(self, img, height):
        """Under the FPS and above the state (the same line when not live)."""
        panel_x, panel_y = PANEL_POS
        line_left = panel_x + SEPARATOR_MARGIN
        line_right = panel_x + PANEL_WIDTH - SEPARATOR_MARGIN
        for offset in (FPS_ROW_HEIGHT, height - STATE_ROW_HEIGHT):
            line_y = panel_y + offset
            cv.line(img, (line_left, line_y), (line_right, line_y), SEPARATOR_COLOR, 1)

    def _draw_gestures(self, img, gestures):
        """One line per possible hand."""
        text_x = PANEL_POS[0] + GESTURE_TEXT_POS[0]
        text_y = PANEL_POS[1] + GESTURE_TEXT_POS[1]
        for i in range(self.max_hands):
            if i < len(gestures):
                line = f"{gestures[i]['name']}: {gestures[i]['score']:.2f}"
            else:
                line = "Hand not showing"
            pos = (text_x, text_y + i * GESTURE_LINE_HEIGHT)
            cv.putText(img, line, pos, FONT, 1, WHITE, 2)

    def _draw_bar(self, img, hold_progress):
        bar_x = PANEL_POS[0] + BAR_X
        bar_y = PANEL_POS[1] + BAR_Y
        cv.rectangle(img, (bar_x, bar_y), (bar_x + BAR_W, bar_y + BAR_H), WHITE, 1)
        cv.rectangle(
            img,
            (bar_x, bar_y),
            (bar_x + int(BAR_W * hold_progress), bar_y + BAR_H),
            GREEN,
            -1,
        )

    def _draw_state(self, img, state, height):
        """Always the last line."""
        pos = (
            PANEL_POS[0] + STATE_LABEL_X,
            PANEL_POS[1] + height - STATE_LABEL_BOTTOM_MARGIN,
        )
        cv.putText(img, state.name, pos, FONT, 1, WHITE, 2)
