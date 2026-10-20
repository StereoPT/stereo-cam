class HoldTimer:
    """Tracks how long a sign has been held, forgiving brief dropouts."""

    def __init__(self, duration, grace):
        """duration: seconds the sign must be held. grace: seconds it may drop
        out without resetting the hold."""
        self.duration = duration
        self.grace = grace
        self.reset()

    def reset(self):
        self.held_since = None
        self.last_seen = None

    def update(self, is_held, now):
        """Feed one frame's result. Returns the hold progress from 0.0 to 1.0."""
        if is_held:
            if self.held_since is None:
                self.held_since = now
            self.last_seen = now
        elif self.held_since is not None and now - self.last_seen > self.grace:
            self.reset()

        if self.held_since is None:
            return 0.0

        return min(1.0, (now - self.held_since) / self.duration)
