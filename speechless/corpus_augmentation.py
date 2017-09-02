__author__ = 'Steffen'

import os

from speechless.labeled_example import LabeledExample, PositionalLabel, LabeledExampleFromFile
from speechless import wavlib
from enum import Enum, auto
from lazy import lazy
from typing import List, Optional
from pathlib import Path
import audioread
import librosa

from speechless.tools import name_without_extension, log


class Augmentation(Enum):
    BackgroundEnvironmental = 'environmental'
    BackgroundSpeech = 'speech'
    BackgroundMusic = 'music'
    Reverb = auto()
    Speed = auto()

class AugmentedLabeledExampleFromFile(LabeledExample):
    def __init__(self,
                 audio_file: Path,
                 augmentation: Augmentation,
                 id: Optional[str] = None,
                 sample_rate_to_convert_to: int = 16000,
                 label: Optional[str] = "nolabel",
                 fourier_window_length: int = 512,
                 hop_length: int = 128,
                 mel_frequency_count: int = 128,
                 label_with_tags: str = None,
                 positional_label: Optional[PositionalLabel] = None):
        # The default values for hop_length and fourier_window_length are powers of 2 near the values specified in the wave2letter paper.

        if id is None:
            id = name_without_extension(audio_file)

        self.audio_file = audio_file
        self.augmentation = augmentation

        super().__init__(
            id=id, get_raw_audio=lambda: self.augment(self),
            label=label, sample_rate=sample_rate_to_convert_to,
            fourier_window_length=fourier_window_length, hop_length=hop_length, mel_frequency_count=mel_frequency_count,
            label_with_tags=label_with_tags, positional_label=positional_label)

    def augment(self):

        if (self.augmentation == Augmentation.BackgroundEnvironmental or
           self.augmentation == Augmentation.BackgroundMusic or
           self.augmentation == Augmentation.BackgroundSpeech):

            background_wav = wavlib.random_wav(self.augmentation)
            output = wavlib.mix_wavs_raw(self.audio_file, background_wav)
            return output

        elif self.augmentation == Augmentation.Reverb:
            return None

        elif self.augmentation == Augmentation.Speed:
            return None

    @property
    def audio_directory(self):
        return Path(self.audio_file.parent)

    @lazy
    def original_sample_rate(self) -> int:
        return LabeledExampleFromFile.file_sample_rate(self.audio_file)

    @staticmethod
    def file_sample_rate(audio_file: Path) -> int:
        with audioread.audio_open(os.path.realpath(str(audio_file))) as input_file:
            return input_file.samplerate

    @lazy
    def duration_in_s(self) -> float:
        try:
            return librosa.get_duration(filename=str(self.audio_file))
        except Exception as e:
            log("Failed to get duration of {}: {}".format(self.audio_file, e))
            return 0

    def sections(self) -> Optional[List[LabeledExample]]:
        if self.positional_label is None:
            return None

        audio = self.get_raw_audio()

        def section(label, start, end):
            return LabeledExample(
                get_raw_audio=lambda: audio[int(start * self.sample_rate):int(end * self.sample_rate)],
                label=label,
                sample_rate=self.sample_rate,
                fourier_window_length=self.fourier_window_length, hop_length=self.hop_length,
                mel_frequency_count=self.mel_frequency_count)

        return [section(label, start, end) for label, (start, end) in self.positional_label.labeled_sections]
