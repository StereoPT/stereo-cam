import tomllib
from dataclasses import dataclass, fields

from .effects import EFFECTS
from .gestures import FINGERTIPS_TOUCHING

# The labels the gesture model can output
GESTURE_NAMES = (
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "ILoveYou",
)

# Everything trigger_gesture can be: the model's labels, plus our own signs
SIGN_NAMES = GESTURE_NAMES + (FINGERTIPS_TOUCHING,)


class ConfigError(Exception):
    """Raised when the config file can't be read or holds something invalid."""


def load_config(path):
    """Read a TOML file into a Config; settings the file leaves out keep
    their defaults. Raises ConfigError for a missing or broken file, or for
    keys that are not settings."""
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"Config file {path} is not valid TOML: {error}") from None

    valid = [field.name for field in fields(Config)]
    unknown = [key for key in data if key not in valid]
    if unknown:
        raise ConfigError(
            f"Unknown setting(s) in {path}: {', '.join(unknown)}. "
            f"Valid settings: {', '.join(valid)}"
        )

    return Config(**data)


@dataclass(frozen=True)
class Config:
    trigger_gesture: str = "FingertipsTouching"
    trigger_threshold: float = 0.8

    hold_duration: float = 0.5
    hold_grace: float = 0.1

    dissolve_duration: float = 1.0
    effect: str = "glitch"  # how you disappear: one of the names in effects.EFFECTS

    requested_fps: int = 30

    camera_index: int = 0

    track_face: bool = True  # find your face, so captions can follow it

    captions: bool = True  # show what you say as subtitles; needs a microphone
    speech_model: str = "base"  # a Whisper model; the first run downloads it
    speech_languages: tuple = ("en", "pt")  # spoken languages; the L key switches
    caption_seconds: float = 4.0  # how long a phrase stays on screen
    caption_scale: float = 1.0  # text size; 1.0 is the base size, 2.0 twice as big
    caption_emphasis: float = (
        1.0  # how much bigger a word you say louder gets; 0 is off
    )
    caption_tilt: float = (
        0.6  # how much the caption tilts with your head; 0 is off, 1 follows fully
    )
    caption_font_seconds: float = 0.4  # how often the caption switches to another font
    microphone: int = -1  # sound input device index; -1 is the system default
    countdown_seconds: int = 3

    def __post_init__(self):
        if self.trigger_gesture not in SIGN_NAMES:
            raise ConfigError(
                f"trigger_gesture {self.trigger_gesture!r} is not a sign. "
                f"Choose one of: {', '.join(SIGN_NAMES)}"
            )
        if not 0 <= self.trigger_threshold <= 1:
            raise ConfigError("trigger_threshold must be between 0 and 1")
        if self.effect not in EFFECTS:
            raise ConfigError(
                f"effect {self.effect!r} is not an effect. "
                f"Choose one of: {', '.join(EFFECTS)}"
            )
        if self.hold_duration <= 0:
            raise ConfigError("hold_duration must be greater than 0")
        if self.dissolve_duration <= 0:
            raise ConfigError("dissolve_duration must be greater than 0")
        if self.hold_grace < 0:
            raise ConfigError("hold_grace can't be negative")
        if self.requested_fps <= 0:
            raise ConfigError("requested_fps must be greater than 0")
        # TOML gives a list; keep it a tuple like the default
        object.__setattr__(self, "speech_languages", tuple(self.speech_languages))
        if not self.speech_languages:
            raise ConfigError("speech_languages needs at least one language")
        if self.speech_model.endswith(".en") and any(
            language != "en" for language in self.speech_languages
        ):
            raise ConfigError(
                f"speech_model {self.speech_model!r} only understands English. "
                "Drop the '.en' to use other languages"
            )
        if self.caption_seconds <= 0:
            raise ConfigError("caption_seconds must be greater than 0")
        if self.caption_scale <= 0:
            raise ConfigError("caption_scale must be greater than 0")
        if self.caption_emphasis < 0:
            raise ConfigError("caption_emphasis can't be negative")
        if self.caption_tilt < 0:
            raise ConfigError("caption_tilt can't be negative")
        if self.caption_font_seconds <= 0:
            raise ConfigError("caption_font_seconds must be greater than 0")
        if self.microphone < -1:
            raise ConfigError("microphone must be -1 (default) or a device index")
        if self.camera_index < 0:
            raise ConfigError("camera_index can't be negative")
        if self.countdown_seconds < 0:
            raise ConfigError("countdown_seconds can't be negative")
