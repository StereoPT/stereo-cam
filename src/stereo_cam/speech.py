import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # what Whisper expects
BLOCK_SECONDS = 0.1
SPEECH_LEVEL = 0.015  # RMS above this counts as speaking
PAUSE_SECONDS = 0.6  # silence that ends an utterance
MAX_UTTERANCE_SECONDS = 8.0  # cut long speech so captions keep arriving
MIN_UTTERANCE_SECONDS = 0.3  # shorter than this is a click or a cough
PARTIAL_SECONDS = 0.7  # how often the phrase so far is re-transcribed while you speak
PRE_ROLL_BLOCKS = (
    3  # blocks kept from before speech starts, so the first word isn't clipped
)


class Transcriber:
    """Listens to the microphone and turns each spoken phrase into text.

    Audio is captured by a sounddevice callback, cut into phrases at pauses,
    and transcribed on a worker thread, so the video loop never waits on it.
    Read the newest phrase with latest()."""

    def __init__(self, model_name, device=None):
        log.info("Loading speech model %r (the first run downloads it)", model_name)
        self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self._device = device
        self._blocks = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._text = ""
        self._heard_at = 0.0
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._stream = None

    def start(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * BLOCK_SECONDS),
            device=self._device,
            callback=self._on_audio,
        )
        self._stream.start()
        self._thread.start()
        log.info("Listening on: %s", sd.query_devices(self._stream.device)["name"])

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def latest(self):
        """The newest phrase and the time.perf_counter() it was heard at."""
        with self._lock:
            return self._text, self._heard_at

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            log.debug("Audio status: %s", status)
        self._blocks.put(indata[:, 0].copy())

    def _work(self):
        pre_roll = []
        utterance = []
        silent_blocks = 0
        pause_blocks = round(PAUSE_SECONDS / BLOCK_SECONDS)
        max_blocks = round(MAX_UTTERANCE_SECONDS / BLOCK_SECONDS)
        partial_blocks = round(PARTIAL_SECONDS / BLOCK_SECONDS)
        since_partial = 0

        while not self._stop.is_set():
            try:
                block = self._blocks.get(timeout=0.2)
            except queue.Empty:
                continue

            speaking = np.sqrt(np.mean(block**2)) > SPEECH_LEVEL

            if not utterance:
                if speaking:
                    utterance = [*pre_roll, block]
                    silent_blocks = 0
                    since_partial = 0
                else:
                    pre_roll = [*pre_roll, block][-PRE_ROLL_BLOCKS:]
                continue

            utterance.append(block)
            silent_blocks = 0 if speaking else silent_blocks + 1
            since_partial += 1

            if silent_blocks >= pause_blocks or len(utterance) >= max_blocks:
                self._transcribe(np.concatenate(utterance))
                utterance = []
                pre_roll = []
            elif since_partial >= partial_blocks and silent_blocks == 0:
                # still talking: show the phrase so far, to be corrected at the pause
                self._transcribe(np.concatenate(utterance))
                since_partial = 0

    def _transcribe(self, audio):
        if len(audio) < MIN_UTTERANCE_SECONDS * SAMPLE_RATE:
            return

        segments, _ = self._model.transcribe(audio, vad_filter=True, beam_size=1)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            return

        log.info("Heard: %s", text)
        with self._lock:
            self._text = text
            self._heard_at = time.perf_counter()
