"""Base classes of this project.

Defines common interfaces, attributes, and utilities for different tasks.
"""

import os
from abc import ABCMeta, abstractmethod

from tensorflow.keras.models import model_from_yaml

from omnizart import MODULE_PATH
from omnizart.utils import get_logger
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

    def _load_model(self, model_path=None, custom_objects=None):
        arch_path, weight_path, conf_path = self._resolve_model_path(model_path)
        model = self._get_model_from_yaml(arch_path, custom_objects=custom_objects)

        try:
            model.load_weights(weight_path)
        except OSError:
            raise FileNotFoundError(
                f"Weight file not found: {weight_path}. Perhaps not yet downloaded?\n"
                "Try execute 'omnizart download-checkpoints'"
            )

        settings = self.setting_class(conf_path=conf_path)
        return model, settings

    def _resolve_model_path(self, model_path=None):
        if model_path in self.settings.checkpoint_path:
            # The given model_path is actually the 'mode'.
            default_path = self.settings.checkpoint_path[model_path]
            model_path = os.path.join(MODULE_PATH, default_path)
            logger.info("Using built-in model %s for transcription.", model_path)
        else:
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

    def _validate_and_get_settings(self, setting_instance):
        if setting_instance is not None:
            assert isinstance(setting_instance, self.setting_class)
            return setting_instance
        return self.settings


class Label:
    """Interface of different label format.

    Plays role for generalize the label format, and subsequent dataset class should
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
    velocity: int
        Velocity of keypress, should be wihtin 0~127
    start_beat: float
        Start beat index of the note.
    end_beat: float
        End beat index of the note.
    note_value: str
        Type of the note (e.g. quater, eighth, sixteenth).
    is_drum: bool
        Whether the note represents the drum channel.
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
