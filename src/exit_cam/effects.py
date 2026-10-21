import functools
import math

import cv2 as cv
import numpy as np

BLOCK_SIZE = 4  # size in pixels of the squares that switch in the pixelate effect
MAX_SHIFT = 40  # pixels the colour channels slide apart at the middle of the glitch
MAX_TEAR = 30  # pixels a torn band of rows can slide sideways at the middle
TEAR_CHANCE = 0.25  # share of bands that tear at the middle of the glitch
BAND_HEIGHTS = (4, 40)  # smallest and biggest height of a torn band, in rows

_rng = np.random.default_rng()


def fade(frame, background, amount):
    """Cross-fade from the frame to the background. amount goes from 0.0 (all
    frame) to 1.0 (all background)."""
    return cv.addWeighted(frame, 1 - amount, background, amount, 0)


@functools.cache
def _noise_map(height, width, block_size):
    """One random number from 0 to 1 per pixel, in square blocks. Cached, so
    the same pattern is used for the whole run."""
    small = np.random.rand(
        math.ceil(height / block_size), math.ceil(width / block_size)
    )
    return cv.resize(small, (width, height), interpolation=cv.INTER_NEAREST)


def pixelate(frame, background, amount):
    """Blocks of the frame switch to the background one by one, in a random
    order. amount goes from 0.0 (all frame) to 1.0 (all background)."""
    height, width = frame.shape[:2]
    switched = _noise_map(height, width, BLOCK_SIZE) < amount
    return np.where(switched[:, :, None], background, frame)


def glitch(frame, background, amount):
    """Tear the picture into sliding bands and split its colour channels, swap
    to the background at the middle, then settle again. It is different on
    every frame, so it flickers. amount goes from 0.0 (all frame) to 1.0 (all
    background)."""
    # 0 at both ends, 1 in the middle
    intensity = 1 - abs(2 * amount - 1)
    source = frame if amount < 0.5 else background
    if intensity * MAX_SHIFT < 1:
        return source

    out = source.copy()
    height = out.shape[0]

    # bands of rows slide sideways, more of them the stronger the glitch
    row = 0
    while row < height:
        band = int(_rng.integers(*BAND_HEIGHTS))
        if _rng.random() < intensity * TEAR_CHANCE:
            slide = int(_rng.uniform(-1, 1) * MAX_TEAR * intensity)
            out[row : row + band] = np.roll(out[row : row + band], slide, axis=1)
        row += band

    # the colour channels drift apart, a little differently every frame
    shift = int(intensity * MAX_SHIFT * _rng.uniform(0.85, 1.0))
    # frames are BGR: blue slides left, red slides right, green stays put
    out[:, :, 0] = np.roll(out[:, :, 0], -shift, axis=1)
    out[:, :, 2] = np.roll(out[:, :, 2], shift, axis=1)
    return out


# Every effect takes (frame, background, amount) and returns a frame
EFFECTS = {
    "fade": fade,
    "pixelate": pixelate,
    "glitch": glitch,
}
