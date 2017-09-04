from contextlib import contextmanager
import array
from sys import byteorder
import wave
import math

import librosa
import numpy as np
import pyaudio

# Microphone stream config
# ------------------------
# This controls how many frames to read each time from mic.
# Talk to David if you want to know why this is set to 128
CHUNK = 128
# The frame size is in bytes (i.e. how many bytes constitute a frame)
FRAME_SIZE = 4
# According to the frame size we pick the right pyaudio format
# Depending on the format each frame consists of a different number of bytes
FORMAT = pyaudio.get_format_from_width(FRAME_SIZE)
CHANNELS = 1
RATE = 16000

RECORD_SECONDS = 3

# MEL spectrogram Settings
# ------------------------
FOURIER_WINDOW_LEN = 512
HOPS = 128


@contextmanager
def demo_pyaudio():
    """Terminates a pyaudio session even if there was an exception."""
    p_master = pyaudio.PyAudio()
    yield p_master
    p_master.terminate()


@contextmanager
def demo_pyaudio_stream(p_master: pyaudio.PyAudio):
    """Closes stream even if there was an exception."""
    stream = p_master.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                           frames_per_buffer=CHUNK)
    yield stream
    stream.close()


def convert_to_ndarray(byteseq):
    chunk_as_array = array.array('f', byteseq)
    if byteorder == 'big':
        chunk_as_array.byteswap()
    return np.asarray(chunk_as_array)


def timed_record_audio(p_master: pyaudio.PyAudio, seconds: int=RECORD_SECONDS):
    chunks_to_record = seconds * RATE / CHUNK
    with demo_pyaudio_stream(p_master) as stream:
        return convert_to_ndarray(record_chunks(stream, chunks_to_record))


def record_chunks(stream: pyaudio.Stream, chunks_to_record: int):
    return b''.join(stream.read(CHUNK) for _ in range(int(chunks_to_record)))


def write2wav(filename: str, audio: np.ndarray) -> None:
    librosa.output.write_wav(filename, audio, sr=RATE)


def _pywav_write(filename: str, audio: bytes) -> None:
    """Write audio bytes to wave file.

    Keeping this for reference only since I discovered (at least on my
    machine), that librosa's wav file output is much much cleaner.

    """
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(FRAME_SIZE)
        wav_file.setframerate(RATE)
        wav_file.writeframes(audio)


def _powerlevel_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """Convert power spectrogram to power level spectrogram.

    Copied from Julius, gotta ask him what the motivation for it is.

    """

    # default value for min_decibel found by experiment (all values except for 0s were above this bound)
    def power_to_decibel(x, min_decibel: float=-150) -> float:
        if x == 0:
            return min_decibel
        l = 10 * math.log10(x)
        return min_decibel if l < min_decibel else l

    return np.vectorize(power_to_decibel)(spectrogram)


def mel_powerlevel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Generate MEL-scaeled power level spectrogram from audio.

    Very much tailored to our specific usecase for now.

    """
    power_spectro = np.abs(librosa.stft(y=audio, n_fft=FOURIER_WINDOW_LEN, hop_length=HOPS))**2
    power_level = _powerlevel_spectrogram(power_spectro)

    mel_basis = librosa.filters.mel(RATE, FOURIER_WINDOW_LEN)
    return mel_basis.dot(power_level)
