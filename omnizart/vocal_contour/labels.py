import abc

import numpy as np
from librosa.core import hz_to_midi

from omnizart.base import Label
from omnizart.constants.midi import LOWEST_MIDI_NOTE


class BaseLabelExtraction(metaclass=abc.ABCMeta):
    """Base class for extract label information.
    Provides basic functions to parse the original label format
    into the target format for training.
    All sub-classes should override the function ``load_label``
    and returns a list of :class:`Label` objects.
    """
    @classmethod
    @abc.abstractmethod
    def load_label(cls, label_path):  # -> list[Label]
        """Load the label file and parse information into ``Label`` class.
        Sub-classes should override this function to process their own label
        format.
        Parameters
        ----------
        label_path: Path
            Path to the label file.
        Returns
        -------
        labels: list[Label]
            List of :class:`Label` instances.
        """
        raise NotImplementedError

    @classmethod
    def extract_label(cls, label_path, t_unit=0.02):
        labels = cls.load_label(label_path)
        fs = round(1 / t_unit)

        max_time = max(label.end_time for label in labels)
        output = np.zeros((round(max_time * fs), 352))
        for label in labels:
            start_idx = round(label.start_time * fs)
            end_idx = round(label.end_time * fs)
            pitch = round((label.note - LOWEST_MIDI_NOTE) * 4)
            output[start_idx:end_idx, pitch] = 1
        return output


class MIR1KlabelExtraction(BaseLabelExtraction):
    """MIR-1K dataset label extraction class."""
    @classmethod
    def load_label(cls, label_path):
        with open(label_path, "r") as lin:
            lines = lin.readlines()

        labels = []
        for idx, line in enumerate(lines):
            note = float(line)
            if note < 0.1:
                # No pitch
                continue
            start_t = 0.01 * idx + 0.02  # The first frame starts from 20ms.
            end_t = start_t + 0.01
            labels.append(Label(start_time=start_t, end_time=end_t, note=note))
        return labels


class MedleyDBLabelExtraction(BaseLabelExtraction):
    """MedleyDB dataset label extraction class."""
    @classmethod
    def load_label(cls, label_path):
        with open(label_path, "r") as fin:
            lines = fin.readlines()

        labels = []
        t_unit = 256 / 44100  # ~= 0.0058 secs
        for line in lines:
            elems = line.strip().split(",")
            sec, hz = float(elems[0]), float(elems[1])  # pylint: disable=invalid-name
            if hz < 1e-10:
                continue
            note = float(hz_to_midi(hz))  # Convert return type of np.float64 to float
            end_t = sec + t_unit
            labels.append(Label(start_time=sec, end_time=end_t, note=note))

        return labels
