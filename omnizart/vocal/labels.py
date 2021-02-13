import abc

import numpy as np

from omnizart.constants import datasets as dset


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
        """Extract SDT label.

        There are 6 types of events as defined in the original paper:
        activation, silence, onset, non-onset, offset, and non-offset.
        The corresponding annotations used in the paper are [a, s, o, o', f, f'].
        The 'activation' includes the onset and offset time. And non-onset and
        non-offset events refer to when there are no onset/offset events.

        Parameters
        ----------
        label_path: Path
            Path to the groun-truth file.
        t_unit: float
            Time unit of each frame.

        Returns
        -------
        sdt_label: 2D numpy array
            Label in SDT format with dimension: Time x 6
        """
        label_list = cls.load_label(label_path)

        max_sec = max([ll.end_time for ll in label_list])
        num_frm = int(max_sec / t_unit) + 10  # Reserve additional 10 frames

        sdt_label = np.zeros((num_frm, 6))
        frm_per_sec = round(1 / t_unit)
        clip = lambda v: np.clip(v, 0, num_frm - 1)
        for label in label_list:
            act_range = range(
                round(label.start_time*frm_per_sec), round(label.end_time*frm_per_sec)  # noqa: E226
            )
            on_range = range(
                round(label.start_time*frm_per_sec - 2), round(label.start_time*frm_per_sec + 4)  # noqa: E226
            )
            off_range = range(
                round(label.end_time*frm_per_sec - 2), round(label.end_time*frm_per_sec + 4)  # noqa: E226
            )
            if len(act_range) == 0:
                continue

            sdt_label[clip(act_range), 0] = 1  # activation
            sdt_label[clip(on_range), 2] = 1  # onset
            sdt_label[clip(off_range), 4] = 1  # offset

        sdt_label[:, 1] = 1 - sdt_label[:, 0]
        sdt_label[:, 3] = 1 - sdt_label[:, 2]
        sdt_label[:, 5] = 1 - sdt_label[:, 4]
        return sdt_label


class CMediaLabelExtraction(BaseLabelExtraction):
    """Label extraction for CMedia dataset."""
    @classmethod
    def load_label(cls, label_path):
        return dset.CMediaStructure.load_label(label_path)


class MIR1KlabelExtraction(BaseLabelExtraction):
    """Label extraction for MIR-1K dataset."""
    @classmethod
    def load_label(cls, label_path):
        return dset.MIR1KStructure.load_label(label_path)


class TonasLabelExtraction(BaseLabelExtraction):
    """Label extraction for TONAS dataset."""
    @classmethod
    def load_label(cls, label_path):
        return dset.TonasStructure.load_label(label_path)
