import logging

log = logging.getLogger(__name__)

MAX_READ_FAILURES = 10  # failed reads in a row before giving up


class CameraError(Exception):
    """Raised when the camera keeps failing to deliver frames."""


class FrameReader:
    """Reads frames from a camera, reusing the last good frame on a failed read.

    Raises CameraError if the very first read fails, or if reads fail
    max_read_failures times in a row.
    """

    def __init__(self, cap, max_read_failures=MAX_READ_FAILURES):
        self.cap = cap
        self.max_read_failures = max_read_failures
        self.failures = 0
        self.last_frame = None

    def read(self):
        ok, frame = self.cap.read()

        if ok:
            self.failures = 0
            self.last_frame = frame
            return frame

        self.failures += 1
        log.warning("Can't receive frame (%d in a row)", self.failures)

        if self.last_frame is None:
            raise CameraError("Camera returned no frames")

        if self.failures >= self.max_read_failures:
            raise CameraError(f"Camera failed {self.failures} reads in a row")

        return self.last_frame
