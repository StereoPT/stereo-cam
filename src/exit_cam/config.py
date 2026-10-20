import tomllib
from dataclasses import dataclass, fields

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
    trigger_gesture: str = "Victory"
    trigger_threshold: float = 0.8

    hold_duration: float = 0.5
    hold_grace: float = 0.1

    dissolve_duration: float = 1.0

    requested_fps: int = 30

    camera_index: int = 0
    countdown_seconds: int = 3

    mask_threshold: int = 25  # how different from the background counts as you

    def __post_init__(self):
        if self.trigger_gesture not in GESTURE_NAMES:
            raise ConfigError(
                f"trigger_gesture {self.trigger_gesture!r} is not a gesture. "
                f"Choose one of: {', '.join(GESTURE_NAMES)}"
            )
        if not 0 <= self.trigger_threshold <= 1:
            raise ConfigError("trigger_threshold must be between 0 and 1")
        if self.hold_duration <= 0:
            raise ConfigError("hold_duration must be greater than 0")
        if self.dissolve_duration <= 0:
            raise ConfigError("dissolve_duration must be greater than 0")
        if not 0 <= self.mask_threshold <= 255:
            raise ConfigError("mask_threshold must be between 0 and 255")
        if self.hold_grace < 0:
            raise ConfigError("hold_grace can't be negative")
        if self.requested_fps <= 0:
            raise ConfigError("requested_fps must be greater than 0")
        if self.camera_index < 0:
            raise ConfigError("camera_index can't be negative")
        if self.countdown_seconds < 0:
            raise ConfigError("countdown_seconds can't be negative")
