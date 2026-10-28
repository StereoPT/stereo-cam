import logging
import math
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
WORD_EDGE = 0.2  # the first and last fifth of a word aren't measured: the timings are rough there
MIN_WORD_SECONDS = 0.05  # a word shorter than this has too little audio to measure
EMPHASIS_FROM_DB = (
    2.5  # louder than the phrase's typical word by this starts to stand out
)
EMPHASIS_FULL_DB = 5.0  # and this much louder is the most emphasis, 1.0
MIN_WORDS_FOR_EMPHASIS = 3  # fewer words than this have nothing to be louder than
PRE_ROLL_BLOCKS = (
    3  # blocks kept from before speech starts, so the first word isn't clipped
)


class Transcriber:
    """Listens to the microphone and turns each spoken phrase into text.

    Audio is captured by a sounddevice callback, cut into phrases at pauses,
    and transcribed on a worker thread, so the video loop never waits on it.
    Read the newest phrase with latest()."""

    def __init__(self, model_name, language, device=None):
        log.info("Loading speech model %r (the first run downloads it)", model_name)
        self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self._device = device
        self.language = language  # what you speak; may be changed while running
        self._blocks = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._words = []  # (word, emphasis) of the newest text
        self._heard_at = 0.0
        self._phrase = 0  # counts the phrases you have started speaking
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

    def set_language(self, language):
        """Switch the spoken language, and drop the caption written in the old one."""
        self.language = language
        with self._lock:
            self._words = []

    def latest(self):
        """The newest words as [(word, emphasis)], the time.perf_counter() they
        were heard at, and which phrase they belong to. Emphasis is 0.0 for a
        word said at your usual loudness, up to 1.0 for one said much louder.
        The words of one phrase grow and get corrected as you speak; the
        number changes when a new phrase starts."""
        with self._lock:
            return self._words, self._heard_at, self._phrase

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
                    with self._lock:
                        self._phrase += 1
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

        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            beam_size=1,
            word_timestamps=True,
        )
        heard = []  # (word, loudness in dB)
        for segment in segments:
            for word in segment.words:
                text = word.word.strip()
                if text:
                    heard.append((text, word_db(audio, word.start, word.end)))
        if not heard:
            return

        words = with_emphasis(heard)
        log.info("Heard: %s", " ".join(word for word, _ in words))
        with self._lock:
            self._words = words
            self._heard_at = time.perf_counter()


def with_emphasis(heard):
    """[(word, emphasis)] for [(word, dB)]. Each word is compared with the
    typical word of its own phrase, so a phrase said loudly as a whole has no
    emphasis, and only a word louder than the rest stands out. Emphasis is
    0.0 up to 1.0."""
    levels = sorted(db for _, db in heard if db is not None)
    if len(levels) < MIN_WORDS_FOR_EMPHASIS:
        return [(word, 0.0) for word, _ in heard]

    typical = levels[len(levels) // 2]  # the median
    words = []
    for word, db in heard:
        emphasis = 0.0
        if db is not None:
            emphasis = (db - typical - EMPHASIS_FROM_DB) / (
                EMPHASIS_FULL_DB - EMPHASIS_FROM_DB
            )
        words.append((word, min(max(emphasis, 0.0), 1.0)))

    return words


def word_db(audio, start, end):
    """Loudness in dB of the part of the audio from start to end seconds, or
    None if that is too short to measure."""
    if end - start < MIN_WORD_SECONDS:
        return None

    trim = (end - start) * WORD_EDGE
    part = audio[
        round((start + trim) * SAMPLE_RATE) : round((end - trim) * SAMPLE_RATE)
    ]
    if len(part) == 0:
        return None

    rms = float(np.sqrt(np.mean(part**2)))
    return 20 * math.log10(rms) if rms > 0 else None
