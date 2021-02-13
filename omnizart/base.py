"""Base classes of this project.

Defines common interfaces, attributes, and utilities for different tasks.
"""

import os
import glob
import random
from os.path import join as jpath
from abc import ABCMeta, abstractmethod

import h5py
import tensorflow as tf
from tensorflow.keras.models import model_from_yaml

from omnizart import MODULE_PATH
from omnizart.utils import get_logger, ensure_path_exists, get_filename
from omnizart.constants.midi import LOWEST_MIDI_NOTE, HIGHEST_MIDI_NOTE


logger = get_logger("Base Class")


class BaseTranscription(metaclass=ABCMeta):
    """Base class of transcription applications."""
    def __init__(self, setting_class, conf_path=None):
        self.setting_class = setting_class
        self.settings = setting_class(conf_path=conf_path)
        self.custom_objects = {}

    @abstractmethod
    def transcribe(self, input_audio, model_path, output="./"):
        raise NotImplementedError

    def get_model(self, settings):
        """Get the model from the python source file.

        This is only for those customized models that can't export *arch.yaml* file,
        and hence need to instanitiate the model from python class, which is not
        that desirable as the architecture is not recorded in a stand-alone file.

        Another way is using model.save() method to export .pb format architecture
        file, but there could be troubles if you want continue to train on these
        models in such format when these are customized models.

        Returns
        -------
        model: tf.keras.Model
            The initialized keras model.
        """
        raise NotImplementedError

    def _load_model(self, model_path=None, custom_objects=None):
        if model_path in self.settings.checkpoint_path:
            # The given model_path is actually the 'transcription_mode'.
            default_path = self.settings.checkpoint_path[model_path]
            model_path = os.path.join(MODULE_PATH, default_path)
            logger.info("Using built-in model %s for transcription.", model_path)

        arch_path, weight_path, conf_path = self._resolve_model_path(model_path)
        settings = self.setting_class(conf_path=conf_path)

        try:
            if not os.path.exists(arch_path):
                model = self.get_model(settings)
                weight_path = weight_path.replace(".h5", "")
                model.load_weights(weight_path).expect_partial()
            else:
                model = self._get_model_from_yaml(arch_path, custom_objects=custom_objects)
                model.load_weights(weight_path)
        except (OSError, tf.python.framework.errors_impl.OpError):
            raise FileNotFoundError(
                f"Checkpoint file not found: {weight_path}. Perhaps not yet downloaded?\n"
                "Try execute 'omnizart download-checkpoints'"
            )

        return model, settings

    def _resolve_model_path(self, model_path=None):
        model_path = os.path.abspath(model_path) if model_path is not None else None
        logger.debug("Absolute path of the given model: %s", model_path)
        if model_path is None:
            default_path = self.settings.checkpoint_path[self.settings.transcription_mode]
            model_path = os.path.join(MODULE_PATH, default_path)
            logger.info("Using built-in model %s for transcription.", model_path)
        elif not os.path.exists(model_path):
            raise FileNotFoundError(f"The given path doesn't exist: {model_path}.")
        elif not os.path.basename(model_path).startswith(self.settings.model.save_prefix.lower()) \
                and not set(["arch.yaml", "weights.h5", "configurations.yaml"]).issubset(os.listdir(model_path)):

            # Search checkpoint folders under the given path
            dirs = [c_dir for c_dir in os.listdir(model_path) if os.path.isdir(c_dir)]
            prefix = self.settings.model.save_prefix.lower()
            cand_dirs = [c_dir for c_dir in dirs if c_dir.startswith(prefix)]

            if len(cand_dirs) == 0:  # pylint: disable=R1720
                raise FileNotFoundError(f"No checkpoint of {prefix} found in {model_path}")
            elif len(cand_dirs) > 1:
                logger.warning("There are multiple checkpoints in the directory. Default to use %s", cand_dirs[0])
            model_path = os.path.join(model_path, cand_dirs[0])

        arch_path = os.path.join(model_path, "arch.yaml")
        weight_path = os.path.join(model_path, "weights.h5")
        conf_path = os.path.join(model_path, "configurations.yaml")

        return arch_path, weight_path, conf_path

    def _get_model_from_yaml(self, arch_path, custom_objects=None):  # pylint: disable=R0201
        return model_from_yaml(open(arch_path, "r").read(), custom_objects=custom_objects)

    def _resolve_feature_output_path(self, dataset_path, settings):  # pylint: disable=R0201
        if settings.dataset.feature_save_path == "+":
            base_output_path = dataset_path
            settings.dataset.save_path = dataset_path
        else:
            base_output_path = settings.dataset.feature_save_path
        train_feat_out_path = jpath(base_output_path, "train_feature")
        test_feat_out_path = jpath(base_output_path, "test_feature")
        ensure_path_exists(train_feat_out_path)
        ensure_path_exists(test_feat_out_path)
        return train_feat_out_path, test_feat_out_path

    def _output_midi(self, output, input_audio, midi=None, verbose=True):
        if output is None:
            return None

        if os.path.isdir(output):
            output = jpath(output, get_filename(input_audio))
        if midi is not None:
            out_path = output if output.endswith(".mid") else f"{output}.mid"
            midi.write(out_path)
            if verbose:
                logger.info("MIDI file has been written to %s.", out_path)
        return output

    def _validate_and_get_settings(self, setting_instance):
        if setting_instance is not None:
            assert isinstance(setting_instance, self.setting_class)
            return setting_instance
        return self.settings


class Label:
    """Interface of different label format.

    Plays the role for generalize the label format, and subsequent dataset class should
    implement functions transforming labels (whether in .mid, .txt, or .csv format)
    and parse the necessary columns into attributes this class holds.

    Parameters
    ----------
    start_time: float
        Onset time of the note in seconds.
    end_time: float
        Offset time of the note in seconds.
    note: int
        Midi number of the number, should be within 21~108.
    instrument: int
        Instrument number in Midi.
    velocity: int
        Velocity of keypress, should be wihtin 0~127
    start_beat: float
        Start beat index of the note.
    end_beat: float
        End beat index of the note.
    note_value: str
        Type of the note (e.g. quater, eighth, sixteenth).
    is_drum: bool
        Whether the note represents the drum note.
    """
    def __init__(
        self,
        start_time,
        end_time,
        note,
        instrument=0,
        velocity=64,
        start_beat=0,
        end_beat=10,
        note_value="",
        is_drum=False
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.note = note
        self.velocity = velocity
        self.instrument = instrument
        self.start_beat = start_beat
        self.end_beat = end_beat
        self.note_value = note_value
        self.is_drum = is_drum

    def __eq__(self, val):
        if not isinstance(val, Label):
            return False

        epsilon = 1e-4  # Tolerance of time difference
        if abs(self.start_time - val.start_time) < epsilon \
                and abs(self.end_time - val.end_time) < epsilon \
                and abs(self.note - val.note) < epsilon \
                and self.velocity == val.velocity \
                and self.instrument == val.instrument \
                and abs(self.start_beat - val.start_beat) < epsilon \
                and abs(self.end_beat - val.end_beat) < epsilon \
                and self.note_value == val.note_value \
                and self.is_drum == val.is_drum:
            return True
        return False

    def __str__(self):
        msg = [
            f"Start time: {self.start_time}",
            f"End time: {self.end_time}",
            f"Note number: {self.note}",
            f"Velocity: {self.velocity}",
            f"Instrument number: {self.instrument}",
            f"Start beat: {self.start_beat}",
            f"End beat: {self.end_beat}",
            f"Note value: {self.note_value}",
            f"Is drum: {self.is_drum}"
        ]
        return ", ".join(msg)

    def __repr__(self):
        return self.__str__()

    @property
    def note(self):
        return self._note

    @note.setter
    def note(self, midi_num):
        if LOWEST_MIDI_NOTE <= midi_num <= HIGHEST_MIDI_NOTE:
            self._note = midi_num
        else:
            logger.warning(
                "The given midi number is out-of-bound and will be skipped. "
                "Received midi number: %d. Available: [%d - %d]",
                midi_num, LOWEST_MIDI_NOTE, HIGHEST_MIDI_NOTE
            )
            self._note = -1

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        assert 0 <= value <= 127
        self._velocity = value


class BaseDatasetLoader:
    """Base dataset loader for yielding training samples.

    Parameters
    ----------
    feature_folder: Path
        Path to the folder that contains the extracted feature files.
    feature_files: list[Path]
        List of paths to the feature files. One of *feature_folder* or *feature_files*
        must be specified.
    num_samples: int
        Total sample number to be yielded.
    slice_hop: int
        Hop size when initializing the start index.
    feat_col_name: str
        Name of the feature column stored in the *.hdf* feature files.

    Yields
    ------
    feature:
        Input feature for training the model.
    label:
        Coressponding label representation.
    """
    def __init__(self, feature_folder=None, feature_files=None, num_samples=100, slice_hop=1, feat_col_name="feature"):
        if feature_files is None:
            assert feature_folder is not None
            self.hdf_files = glob.glob(f"{feature_folder}/*.hdf")
        else:
            self.hdf_files = feature_files

        if len(self.hdf_files) == 0:
            logger.warning("Warning! No feature file was found in the given path.")

        self.slice_hop = slice_hop
        self.feat_col_name = feat_col_name

        self.hdf_refs = {}
        for hdf in self.hdf_files:
            try:
                self.hdf_refs[hdf] = h5py.File(hdf, "r")
            except OSError:
                msg = f"Resource temporarily unavailable due to file being opened without closing. Resource: {hdf}"
                logger.error(msg)
                raise OSError(msg)
        self.num_samples = num_samples

        # Initialize indices of index-to-file mapping to ensure all samples
        # will be visited during training.
        length_map = {hdf: len(hdf_ref[feat_col_name]) for hdf, hdf_ref in self.hdf_refs.items()}
        self.total_length = sum(length_map.values())
        self.start_idxs = list(range(0, self.total_length, slice_hop))
        self.idx_to_hdf_map = {}
        cur_len = 0
        cur_iid = 0
        for hdf, length in length_map.items():
            end_iid = length // slice_hop
            for iid in range(cur_iid, cur_iid+end_iid):  # noqa: E226
                start_idx = self.start_idxs[iid]
                self.idx_to_hdf_map[start_idx] = (hdf, start_idx - cur_len)
            cur_len += end_iid * slice_hop
            cur_iid += end_iid
        diff = set(self.start_idxs) - set(self.idx_to_hdf_map.keys())
        self.cut_idx = len(diff)
        if self.cut_idx > 0:
            self.start_idxs = self.start_idxs[:-self.cut_idx]
        random.shuffle(self.start_idxs)

        logger.info("Total samples: %s", len(self.start_idxs))

    def __iter__(self):
        for _ in range(self.num_samples):
            if len(self.start_idxs) == 0:
                # Shuffle the indexes after visiting all the samples in the dataset.
                self.start_idxs = list(range(0, self.total_length, self.slice_hop))
                if self.cut_idx > 0:
                    self.start_idxs = self.start_idxs[:-self.cut_idx]
                random.shuffle(self.start_idxs)

            start_idx = self.start_idxs.pop()
            hdf_name, slice_start = self.idx_to_hdf_map[start_idx]

            feat = self._get_feature(hdf_name, slice_start)
            label = self._get_label(hdf_name, slice_start)
            feat, label = self._pre_yield(feat, label)

            yield feat, label

    def _get_feature(self, hdf_name, slice_start):
        return self.hdf_refs[hdf_name][self.feat_col_name][slice_start:slice_start + self.slice_hop].squeeze()

    def _get_label(self, hdf_name, slice_start):
        return self.hdf_refs[hdf_name]["label"][slice_start:slice_start + self.slice_hop].squeeze()

    def _pre_yield(self, feature, label):
        return feature, label

    def get_dataset(self, batch_size, output_types=None, output_shapes=None):
        def gen_wrapper():
            for data in self:
                yield data

        return tf.data.Dataset.from_generator(
                gen_wrapper, output_types=output_types, output_shapes=output_shapes
            ) \
            .batch(batch_size, drop_remainder=True) \
            .prefetch(tf.data.experimental.AUTOTUNE)
