import os
import abc

import numpy as np

from omnizart.constants import datasets as dset
from omnizart.constants.midi import MUSICNET_INSTRUMENT_PROGRAMS, LOWEST_MIDI_NOTE
from omnizart.io import dump_pickle
from omnizart.utils import get_logger


logger = get_logger("Music Labels")


class LabelType:
    """Defines different types of `music` label for training.

    Defines functions that converts the customized label format into numpy
    array. With the customized format, it is more flexible to transform
    labels into different different numpy formats according to the usage
    scenario, and also saves a lot of storage space by using the customized
    format.

    Parameters
    ----------
    mode: ['note', 'note-stream', 'pop-note-stream', 'frame', 'frame-stream']
        Mode of label conversion.

        * note: outputs onset and duration channel
        * note-stream: outputs onset and duration channel of instruments (for MusicNet)
        * pop-note-stream: similar to ``note-stream`` mode, but is for ``Pop`` dataset
        * frame: same as ``note`` mode. To truely output duration channel only, use \
        `true-frame` mode.
        * frame-stream: same as ``note-stream``. To truely output duration channel only \
        for each instrument, use ``true-frame-stream`` mode.
    """
    def __init__(self, mode):
        self.mode = mode

        self._classical_channel_mapping = self._init_classical_channel_mapping()
        self._pop_channel_mapping = self._init_pop_channel_mapping()
        self._note_channel_mapping = {i: 1 for i in range(128)}
        self._classical_midi_types = len(set(self._classical_channel_mapping.values()))
        self._pop_midi_types = len(set(self._pop_channel_mapping.values()))

        self.mode_mapping = {
            "true-frame": {"conversion_func": self.get_frame, "out_classes": 2},
            "frame": {"conversion_func": self.get_frame_onset, "out_classes": 3},
            "note": {"conversion_func": self.get_frame_onset, "out_classes": 3},
            "true-frame-stream": {"conversion_func": self.multi_inst_frm, "out_classes": 12},
            "frame-stream": {"conversion_func": self.multi_inst_note, "out_classes": 23},
            "note-stream": {"conversion_func": self.multi_inst_note, "out_classes": 23},
            "pop-note-stream": {"conversion_func": self.multi_pop_note, "out_classes": 13}
        }
        if mode not in self.mode_mapping:
            raise ValueError(f"Available mode: {self.mode_mapping.keys()}. Provided: {mode}")

    def _init_classical_channel_mapping(self):  # pylint: disable=R0201
        return {val: idx + 1 for idx, val in enumerate(MUSICNET_INSTRUMENT_PROGRAMS)}

    def _init_pop_channel_mapping(self):  # pylint: disable=R0201
        guitar = {i: 1 for i in range(24, 32)}
        bass = {i: 2 for i in range(32, 40)}
        strings = {i: 3 for i in range(40, 56)}
        organ = {i: 4 for i in range(16, 24)}
        piano = {i: 5 for i in range(8)}
        others = {i: 6 for i in range(128)}
        return {**others, **guitar, **bass, **strings, **organ, **piano}

    def get_available_modes(self):
        return list(self.mode_mapping.keys())

    def get_conversion_func(self):
        return self.mode_mapping[self.mode]["conversion_func"]

    def get_out_classes(self):
        return self.mode_mapping[self.mode]["out_classes"]

    def get_frame(self, label):
        frame = label_conversion(label, channel_mapping=self._classical_channel_mapping, mpe=True)
        off = 1.0 - frame
        off = np.expand_dims(off, 2)
        return np.dstack([off, frame])

    def get_frame_onset(self, label):
        frame = self.get_frame(label)
        onset = label_conversion(
            label, channel_mapping=self._note_channel_mapping, onsets=True, mpe=True
        )

        frame[:, :, 1] -= onset
        frm_on = np.dstack([frame, onset])
        frm_on[:, :, 0] = 1 - np.sum(frm_on[:, :, 1:], axis=2)

        return frm_on

    def multi_inst_frm(self, label):
        frame = label_conversion(label, channel_mapping=self._classical_channel_mapping)
        off = 1.0 - np.sum(frame, axis=2)
        off = np.expand_dims(off, 2)
        return np.dstack([off, frame])

    def multi_inst_note(self, label):
        onsets = label_conversion(label, channel_mapping=self._classical_channel_mapping, onsets=True)
        dura = label_conversion(label, channel_mapping=self._classical_channel_mapping) - onsets
        out = np.zeros(onsets.shape[:-1] + (23,))

        for i in range(self._classical_midi_types):
            out[:, :, i*2+2] = onsets[:, :, i]  # noqa: E226
            out[:, :, i*2+1] = dura[:, :, i]  # noqa: E226
        out[:, :, 0] = 1 - np.sum(out[:, :, 1:], axis=2)

        return out

    def multi_pop_note(self, label):
        onsets = label_conversion(
            label, onsets=True, channel_mapping=self._pop_channel_mapping
        )
        dura = label_conversion(
            label, channel_mapping=self._pop_channel_mapping
        ) - onsets
        out = np.zeros(onsets.shape[:-1] + (13,))

        for i in range(self._pop_midi_types):
            out[:, :, i*2+2] = onsets[:, :, i]  # noqa: E226
            out[:, :, i*2+1] = dura[:, :, i]  # noqa: E226
        out[:, :, 0] = 1 - np.sum(out[:, :, 1:], axis=2)

        return out


def label_conversion(
    label,
    ori_feature_size=352,
    feature_num=352,
    base=88,
    mpe=False,
    onsets=False,
    channel_mapping=None
):
    """Converts the customized label format into numpy array.

    Parameters
    ----------
    label: object
        List of dict that is in customized label format.
    ori_feature_size: int
        Size of the original feature dimension.
    feature_num: int
        Size of the target output feature dimension.
    base: int
        Number of total available pitches.
    mpe: bool
        Whether to merge all channels into a single one, discarding information
        about instruments.
    onsets: bool
        Fill in onset probabilities if set to true, or fill one to all activations.
    channel_mapping: dict
        Maps the instrument program number to the specified channel index, used
        to indicate which channel should represent what instruments.

    See Also
    --------
    omnizart.music.labels.BaseLabelExtraction.extract_label:
        Function that generates the customized label format.
    """
    assert ori_feature_size % base == 0
    scale = ori_feature_size // base

    if channel_mapping is None:
        channel_mapping = {i: i+1 for i in range(128)}  # noqa: E226

    inst_num = len(set(channel_mapping.values()))
    output = np.zeros((len(label), ori_feature_size, inst_num))  # noqa: E226
    for t, lab in enumerate(label):
        if len(lab) == 0:
            continue

        for pitch, insts in lab.items():
            for inst, prob in insts.items():
                inst = int(inst)
                if inst not in channel_mapping:
                    continue

                pitch = int(pitch)
                channel = channel_mapping[inst] - 1
                output[t, pitch*scale:(pitch+1)*scale, channel] = prob  # noqa: E226

    if not onsets:
        output[output>0] = 1  # noqa: E225

    pad_b = (feature_num - output.shape[1]) // 2
    pad_t = feature_num - pad_b - output.shape[1]
    if pad_b > 0 or pad_t > 0:
        output = np.pad(output, ((0, 0), (pad_b, pad_t), (0, 0)), constant_values=0)

    if mpe:
        output = np.nanmax(output, axis=2)

    return output


class BaseLabelExtraction(metaclass=abc.ABCMeta):
    """Base class for extract label informations.

    Provides basic functions to process native label format into the format
    required by ``music`` module. All sub-classes should parse the original
    label information into :class:`Label` class.

    See Also
    --------
    omnizart.music.labels.label_conversion:

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
    def process(cls, label_list, out_path, t_unit=0.02, onset_len_sec=0.05):
        """Process the given list of label files and output to the target folder.

        Parameters
        ----------
        label_list: list[Path]
            List of label paths.
        out_path: Path
            Path for saving the extracted label files.
        t_unit: float
            Time unit of each step in seconds. Should be consistent with the time unit of
            each frame of the extracted feature.
        onset_len_sec: float
            Length of the first few frames with probability one. The later onset
            probabilities will be in a 'fade-out' manner until the note offset.
        """
        for idx, label_path in enumerate(label_list):
            print(f"Progress: {idx+1}/{len(label_list)} - {label_path}" + " "*6, end="\r")  # noqa: E226
            label_obj = cls.extract_label(label_path, t_unit=t_unit, onset_len_sec=onset_len_sec)
            basename = os.path.basename(label_path)  # File name with extension
            filename, _ = os.path.splitext(basename)  # File name without extension
            output_name = cls.name_transform(filename)  # Output the same name as feature file
            output_path = os.path.join(out_path, output_name + ".pickle")
            dump_pickle(label_obj, output_path)
        print("")

    @classmethod
    def extract_label(cls, label_path, t_unit, onset_len_sec=0.05):
        """Extract labels into customized storage format.

        Process the given path of label into list of :class:`Label` instances,
        then further convert them into deliberately customized storage format.

        Parameters
        ----------
        label_path: Path
            Path to the label file.
        t_unit: float
            Time unit of each step in seconds. Should be consistent with the time unit of
            each frame of the extracted feature.
        onset_len_sec: float
            Length of the first few frames with probability one. The later onset
            probabilities will be in a 'fade-out' manner until the note offset.
        """
        label_list = cls.load_label(label_path)

        end_note = max(label_list, key=lambda label: label.end_time)
        num_frm = int(round(end_note.end_time / t_unit))
        label_obj = [{} for _ in range(num_frm)]
        for label in label_list:
            start_frm = int(round(label.start_time / t_unit))
            end_frm = int(round(label.end_time / t_unit))
            pitch = str(label.note - LOWEST_MIDI_NOTE)
            onset_value = 1
            onset_len_frm = int(round(onset_len_sec / t_unit))
            for idx, frm_idx in enumerate(range(start_frm, end_frm)):
                if pitch not in label_obj[frm_idx]:
                    label_obj[frm_idx][pitch] = {}
                label_obj[frm_idx][pitch][str(label.instrument)] = onset_value

                # Decrease the onset probability
                if idx >= onset_len_frm and onset_value > 1e-5:
                    onset_value /= idx
        return label_obj

    @classmethod
    def name_transform(cls, name):
        """Maps the filename of label to the same name of the corresponding wav file.

        Parameters
        ----------
        name: str
            Name of the label file, without parent directory prefix and file extension.

        Returns
        -------
        trans_name: str
            The name same as the coressponding wav (or says feature) file.
        """
        return name


class MaestroLabelExtraction(BaseLabelExtraction):
    """Label extraction class for Maestro dataset"""
    @classmethod
    def load_label(cls, label_path):
        return dset.MaestroStructure.load_label(label_path)


class MapsLabelExtraction(BaseLabelExtraction):
    """Label extraction class for Maps dataset"""
    @classmethod
    def load_label(cls, label_path):
        return dset.MapsStructure.load_label(label_path)


class MusicNetLabelExtraction(BaseLabelExtraction):
    """Label extraction class for MusicNet dataset"""
    @classmethod
    def load_label(cls, label_path):
        return dset.MusicNetStructure.load_label(label_path)


class SuLabelExtraction(MaestroLabelExtraction):
    """Label extraction class for Extended-Su dataset

    Uses the same process as Maestro dataset
    """


class PopLabelExtraction(MaestroLabelExtraction):
    """Label extraction class for Pop Rhythm dataset"""
    @classmethod
    def name_transform(cls, name):
        return name.replace("align_mid", "ytd_audio")
