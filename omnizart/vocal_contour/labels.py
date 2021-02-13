import abc

import numpy as np

from omnizart.constants import datasets as dset
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
        return dset.MIR1KStructure.load_label(label_path)


class MedleyDBLabelExtraction(BaseLabelExtraction):
    """MedleyDB dataset label extraction class."""
    @classmethod
    def load_label(cls, label_path):
        return dset.MedleyDBStructure.load_label(label_path)
