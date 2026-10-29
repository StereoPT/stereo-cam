import functools
import math

import cv2 as cv
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections
from PIL import Image, ImageDraw, ImageFilter

from .fonts import scaled

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

# The caption sizes below are for capitals CAPTION_CAP_HEIGHT pixels tall, and
# grow or shrink with the text.
CAPTION_CAP_HEIGHT = 26
CAPTION_LINE_HEIGHT = 46
CAPTION_BASELINE_UP = 12  # how far the baseline sits above the bottom of a line
CAPTION_MARGIN = 30  # from the bottom of the frame, and the most width given to the text on each side
CAPTION_PAD = 12  # around the text, room for its shadow
CAPTION_GAP = 10  # between the face box and a caption beside it
CAPTION_MIN_WIDTH = 220  # narrowest a caption beside the face is allowed to get
CAPTION_OUTLINE = 1  # pixels
CAPTION_OUTLINE_COLOR = (0, 0, 0, 255)
CAPTION_TEXT_COLOR = (255, 255, 255, 255)
CAPTION_SHADOW_OFFSET = (3, 3)
CAPTION_SHADOW_BLUR = 2
CAPTION_SHADOW_COLOR = (0, 0, 0, 170)


def draw_centered_text(img, text, scale, thickness):
    (text_width, text_height), _ = cv.getTextSize(text, FONT, scale, thickness)
    x = (img.shape[1] - text_width) // 2
    y = (img.shape[0] + text_height) // 2
    cv.putText(img, text, (x, y), FONT, scale, WHITE, thickness)


def _line_width(line, font):
    """Width in pixels of [(word, emphasis, pop, alpha)] drawn in a row; an
    emphasised word is drawn bigger, so it takes more room."""
    space = font.getlength(" ")
    words = [scaled(font, word[1]).getlength(word[0]) for word in line]
    return sum(words) + space * (len(words) - 1)


def _wrap(words, font, max_width):
    """Split [(word, emphasis, pop, alpha)] into lines that fit max_width
    pixels in the given font."""
    lines = []
    line = []
    for word in words:
        if line and _line_width([*line, word], font) > max_width:
            lines.append(line)
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(line)

    return lines


def _line_height(line, scale):
    """A line with emphasised words is taller, as they grow upward."""
    biggest = max(word[1] for word in line)
    return round(CAPTION_LINE_HEIGHT * scale * max(biggest, 1.0))


def _over(layer, image, x, y):
    """Draw image over the RGBA layer with its top-left corner at (x, y),
    cutting off whatever falls outside the layer."""
    left, top = max(x, 0), max(y, 0)
    right = min(x + image.width, layer.width)
    bottom = min(y + image.height, layer.height)
    if right <= left or bottom <= top:
        return

    part = image.crop((left - x, top - y, right - x, bottom - y))
    layer.alpha_composite(part, (left, top))


def _draw_word(layer, word, font, x, baseline, pop_scale, alpha, outline, room):
    """One word with its left edge at x. A word at full size and opacity is
    drawn straight on the layer; any other is drawn on its own, resized
    around its baseline (so it grows upward) and faded. room is how many
    pixels above the baseline the word may need."""
    style = {
        "font": font,
        "anchor": "ls",
        "fill": CAPTION_TEXT_COLOR,
        "stroke_width": outline,
        "stroke_fill": CAPTION_OUTLINE_COLOR,
    }
    if pop_scale == 1.0 and alpha == 1.0:
        ImageDraw.Draw(layer).text((x, baseline), word, **style)
        return

    margin = round(font.size * 0.4)
    width = round(font.getlength(word)) + 2 * margin
    own = Image.new("RGBA", (width, room + margin), (0, 0, 0, 0))
    ImageDraw.Draw(own).text((margin, room), word, **style)
    if alpha < 1.0:
        own.putalpha(own.getchannel("A").point(lambda a: round(a * alpha)))
    if pop_scale != 1.0:
        own = own.resize(
            (
                max(round(own.width * pop_scale), 1),
                max(round(own.height * pop_scale), 1),
            ),
            Image.Resampling.BICUBIC,
        )

    # the word's centre and baseline stay where the full-size word's would be
    centre = x - margin + width / 2
    _over(
        layer,
        own,
        round(centre - own.width / 2),
        round(baseline - room * pop_scale),
    )


def _caption_layer(lines, font, width, height, scale, fade):
    """The caption as an RGBA image: a soft shadow under outlined text."""
    pad = round(CAPTION_PAD * scale)
    outline = max(CAPTION_OUTLINE, round(CAPTION_OUTLINE * scale))
    text = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    space = font.getlength(" ")
    bottom = pad
    for line in lines:
        # lines share a baseline position, so switching fonts doesn't make them jump
        bottom += _line_height(line, scale)
        baseline = bottom - round(CAPTION_BASELINE_UP * scale)
        x = float(pad)
        for word, emphasis, pop_scale, alpha in line:
            word_font = scaled(font, emphasis)
            room = round(CAPTION_LINE_HEIGHT * scale * max(emphasis, 1.0))
            _draw_word(
                text,
                word,
                word_font,
                round(x),
                baseline,
                pop_scale,
                alpha,
                outline,
                room,
            )
            x += word_font.getlength(word) + space

    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow.putalpha(
        text.getchannel("A").point(lambda a: CAPTION_SHADOW_COLOR[3] * a // 255)
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(CAPTION_SHADOW_BLUR * scale))
    shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow_layer.paste(
        Image.new("RGBA", (width, height), CAPTION_SHADOW_COLOR[:3] + (255,)),
        (
            round(CAPTION_SHADOW_OFFSET[0] * scale),
            round(CAPTION_SHADOW_OFFSET[1] * scale),
        ),
        shadow.getchannel("A"),
    )

    layer = Image.alpha_composite(shadow_layer, text)
    if fade < 1.0:
        layer.putalpha(layer.getchannel("A").point(lambda a: round(a * fade)))

    return layer


def draw_caption(img, words, font, face=None, scale=1.0, fade=1.0, tilt=0.0):
    """Subtitles: outlined text with a soft shadow, no box. words is
    [(word, emphasis, pop, alpha)]: emphasis is how many times bigger the word
    is drawn (1.0 is normal; a word said loudly gets more), pop is the extra
    size of its pop-in, and alpha how see-through it is. Beside the face, from
    its top-right corner (the caption's bottom edge level with the top of the
    face box), when a face box (x, y, width, height) is given; otherwise at
    the bottom of the frame. Shows the last lines if the text is too long to
    fit. scale is how much bigger than the default size the font is, so the
    spacing grows with it. fade is the opacity of the whole caption. tilt is
    an angle in radians, clockwise, that the caption is turned by around the
    middle of the face, as if it were attached to your head."""
    height, width = img.shape[:2]
    pad = round(CAPTION_PAD * scale)
    if face is None:
        max_text_width = width - 2 * CAPTION_MARGIN
    else:
        face_x, face_y, face_width, face_height = face
        room = width - (face_x + face_width) - CAPTION_MARGIN
        max_text_width = max(room, round(CAPTION_MIN_WIDTH * scale)) - 2 * pad

    lines = _wrap(words, font, max_text_width)[-3:]
    if not lines:
        return

    box_height = sum(_line_height(line, scale) for line in lines) + 2 * pad
    box_width = round(max(_line_width(line, font) for line in lines)) + 2 * pad
    if face is None:
        left = (width - box_width) // 2
        top = height - CAPTION_MARGIN - box_height
    else:
        # keep it inside the frame, even when that means overlapping the face
        left = min(face_x + face_width + round(CAPTION_GAP * scale), width - box_width)
        top = min(face_y - box_height, height - box_height)

    left, top = max(left, 0), max(top, 0)
    layer = _caption_layer(lines, font, box_width, box_height, scale, fade)
    if face is not None and tilt:
        left, top, layer = _tilted(
            layer,
            (left + box_width / 2, top + box_height / 2),
            (face_x + face_width / 2, face_y + face_height / 2),
            tilt,
            (width, height),
        )

    _blend(img, np.asarray(layer), left, top)


def _tilted(layer, centre, pivot, angle, frame_size):
    """The caption layer turned clockwise by angle (radians) around pivot, as
    (left, top, layer): where its top-left corner goes, and the larger image
    that holds the turned layer. It is moved back inside a frame_size (width,
    height) frame if turning would take it out."""
    cos, sin = math.cos(angle), math.sin(angle)
    dx, dy = centre[0] - pivot[0], centre[1] - pivot[1]
    new_x = pivot[0] + dx * cos - dy * sin
    new_y = pivot[1] + dx * sin + dy * cos
    turned = layer.rotate(-math.degrees(angle), Image.Resampling.BICUBIC, expand=True)
    left = round(new_x - turned.width / 2)
    top = round(new_y - turned.height / 2)
    # turning swings the caption outward; keep all of it in the frame
    left = min(max(left, 0), max(frame_size[0] - turned.width, 0))
    top = min(max(top, 0), max(frame_size[1] - turned.height, 0))
    return left, top, turned


def _blend(img, layer, left, top):
    """Draw an RGBA array over the BGR image with its top-left corner at
    (left, top), cutting off whatever falls outside the image."""
    x0, y0 = max(left, 0), max(top, 0)
    x1 = min(left + layer.shape[1], img.shape[1])
    y1 = min(top + layer.shape[0], img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return

    part = layer[y0 - top : y1 - top, x0 - left : x1 - left]
    area = img[y0:y1, x0:x1]
    alpha = part[:, :, 3:4].astype(np.float32) / 255
    colour = part[:, :, 2::-1]  # RGB to BGR
    area[:] = area * (1 - alpha) + colour * alpha


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
