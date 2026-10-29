import functools
import logging
import random
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import ImageFont

log = logging.getLogger(__name__)

# your own fonts; not part of the repository
FONT_DIR = Path(__file__).parent / "fonts"
# open-licence fonts that come with it, used when FONT_DIR has none
DEFAULT_FONT_DIR = Path(__file__).parent / "default_fonts"
FONT_SUFFIXES = (".ttf", ".otf")
MEASURE_SIZE = 100  # size the fonts are measured at to work out their scale


def _font_files(folder):
    """The font files in a folder, or none if it doesn't exist."""
    if not folder.is_dir():
        return []

    return sorted(p for p in folder.iterdir() if p.suffix.lower() in FONT_SUFFIXES)


@functools.cache
def font_paths():
    """Every font file in FONT_DIR as (path, size_per_cap, characters).
    size_per_cap is the font size at which a capital letter is one pixel tall:
    fonts at the same point size look very different in size, so each is sized
    by its capitals instead. characters are the code points the font can draw."""
    files = _font_files(FONT_DIR)
    if files:
        log.info("Using %d fonts from %s", len(files), FONT_DIR)
    else:
        files = _font_files(DEFAULT_FONT_DIR)
        log.info("No fonts in %s, using the %d default ones", FONT_DIR, len(files))

    fonts = []
    for path in files:
        measure = ImageFont.truetype(str(path), MEASURE_SIZE)
        _, top, _, bottom = measure.getbbox("H")
        cap = bottom - top
        if cap <= 0:
            continue

        with TTFont(path, lazy=True) as font_file:
            characters = frozenset(font_file.getBestCmap())

        fonts.append((str(path), MEASURE_SIZE / cap, characters))

    if not fonts:
        raise FileNotFoundError(f"No .ttf or .otf fonts found in {DEFAULT_FONT_DIR}")

    return fonts


@functools.lru_cache(maxsize=128)
def load_font(path, size_per_cap, cap_height):
    return ImageFont.truetype(path, round(size_per_cap * cap_height))


@functools.lru_cache(maxsize=256)
def _variant(path, size):
    return ImageFont.truetype(path, size)


def scaled(font, factor):
    """The same font drawn `factor` times as big."""
    if factor == 1.0:
        return font

    return _variant(font.path, max(round(font.size * factor), 1))


class FontRotation:
    """Picks the caption font: a random one, changed every `seconds`, never
    the same twice in a row."""

    def __init__(self, seconds):
        self.seconds = seconds
        self._fonts = font_paths()
        self._index = random.randrange(len(self._fonts))
        self._changed_at = None

    def current(self, now, cap_height, text=""):
        """The current font, sized so its capital letters are cap_height pixels.
        It can draw every letter of text: a font without, say, the accents a
        caption needs is skipped."""
        needed = {ord(c) for c in text if ord(c) > 127}
        usable = [i for i, font in enumerate(self._fonts) if needed <= font[2]]
        if not usable:
            usable = list(range(len(self._fonts)))  # none can; draw it anyway

        if self._changed_at is None:
            self._changed_at = now
        if self._index not in usable:
            self._index = random.choice(usable)
            self._changed_at = now
        elif now - self._changed_at >= self.seconds:
            others = [i for i in usable if i != self._index]
            if others:
                self._index = random.choice(others)
                self._changed_at = now

        path, size_per_cap, _ = self._fonts[self._index]
        return load_font(path, size_per_cap, cap_height)
