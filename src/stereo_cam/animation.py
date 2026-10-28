POP_SECONDS = 0.25  # how long a new word takes to pop in
FADE_IN_SECONDS = 0.12  # a new word goes from invisible to solid in this time
FADE_OUT_SECONDS = 0.5  # the whole caption fades out over its last moments
OVERSHOOT = 1.9  # how far past its size a new word bounces; 0 is no bounce
MIN_SCALE = 0.05  # a word is never scaled to nothing


def pop(age):
    """Scale and opacity of a word that appeared `age` seconds ago: it grows
    from its baseline with a small bounce, and fades in."""
    if age >= POP_SECONDS:
        return 1.0, 1.0

    t = max(age, 0) / POP_SECONDS
    # ease-out-back: overshoots 1.0, then settles
    scale = 1 + (OVERSHOOT + 1) * (t - 1) ** 3 + OVERSHOOT * (t - 1) ** 2
    return max(scale, MIN_SCALE), min(age / FADE_IN_SECONDS, 1.0)


def fade_out(now, heard_at, caption_seconds):
    """Opacity of the whole caption: solid, then fading to nothing as it nears
    the end of caption_seconds."""
    fade = min(FADE_OUT_SECONDS, caption_seconds)
    remaining = heard_at + caption_seconds - now
    return min(max(remaining / fade, 0.0), 1.0)


EMPHASIS_EASE = (
    10.0  # how quickly a word grows to (or shrinks from) its emphasis, per second
)


class WordTracker:
    """Remembers when each word of the caption first appeared, and eases its
    emphasis. The speech model rewrites words while you talk, so a word keeps
    its time if it is still there at the same position, and only words added
    at the end are new. A new phrase starts over."""

    def __init__(self):
        self._phrase = None
        self._last = None
        self._words = []  # [appeared_at, emphasis shown]

    def update(self, words, phrase, now):
        """Take the current words as [(word, emphasis)]; returns
        [(word, age in seconds, emphasis to show)]."""
        if phrase != self._phrase:
            self._phrase = phrase
            self._words = []

        dt = 0.0 if self._last is None else now - self._last
        self._last = now
        ease = min(dt * EMPHASIS_EASE, 1.0)

        shown = []
        for i, (_, emphasis) in enumerate(words):
            if i < len(self._words):
                appeared, current = self._words[i]
                shown.append([appeared, current + (emphasis - current) * ease])
            else:
                shown.append([now, 0.0])
        self._words = shown

        return [
            (word, now - appeared, current)
            for (word, _), (appeared, current) in zip(words, shown, strict=True)
        ]


class Spring:
    """A point pulled toward a moving target by a spring, so it trails the
    target and settles with a little overshoot, instead of tracking it
    exactly. frequency is how quickly it responds (radians per second);
    damping is 1.0 for no overshoot, and lower for a bouncier settle."""

    MAX_STEP = 0.05  # seconds; a longer pause (a lag spike) is treated as this

    def __init__(self, frequency, damping):
        self.frequency = frequency
        self.damping = damping
        self.position = None
        self._velocity = None

    def reset(self):
        """Forget where it was: the next update starts right at the target."""
        self.position = None

    def update(self, target, dt):
        """Move toward target (a tuple of numbers) for dt seconds; returns the
        new position."""
        if self.position is None:
            self.position = list(target)
            self._velocity = [0.0] * len(target)
            return tuple(self.position)

        dt = min(dt, self.MAX_STEP)
        stiffness = self.frequency**2
        friction = 2 * self.damping * self.frequency
        for i, goal in enumerate(target):
            pull = stiffness * (goal - self.position[i]) - friction * self._velocity[i]
            self._velocity[i] += pull * dt
            self.position[i] += self._velocity[i] * dt

        return tuple(self.position)
