import os
import yaml
from abc import ABCMeta, abstractmethod

from tensorflow.keras.models import model_from_yaml

import omnizart
from omnizart.utils import load_yaml, get_logger


MODULE_PATH = os.path.abspath(omnizart.__file__+"/..")
SETTING_DIR = os.path.join(MODULE_PATH, "defaults")

logger = get_logger("Base Class")


class BaseTranscription(metaclass=ABCMeta):
    def __init__(self, setting_class):
        self.setting_class = setting_class

        default_conf_path = os.path.join(SETTING_DIR, setting_class.default_setting_file)
        logger.info("Loading default configurations: %s", default_conf_path)
        self.settings = self._load_settings(default_conf_path)

    @abstractmethod
    def transcribe(self, input_audio, model_path, output="./"):
        raise NotImplementedError

    def _load_model(self, model_path=None, custom_objects=None):
        arch_path, weight_path, conf_path = self._resolve_model_path(model_path)
        model = self._get_model_from_yaml(arch_path, custom_objects=custom_objects)
        model.load_weights(weight_path)
        settings = self._load_settings(conf_path)
        return model, settings

    def _load_settings(self, setting_path):
        json_obj = load_yaml(setting_path)
        settings = self.setting_class()
        settings.from_json(json_obj)
        return settings

    def _resolve_model_path(self, model_path=None):
        if model_path is None:
            default_path = self.settings.checkpoint_path[self.settings.transcription_mode]
            model_path = os.path.join(MODULE_PATH, default_path)
            logger.info("Using built-in model %s for transcription.", model_path)
        elif not os.path.exists(model_path):
            raise FileNotFoundError(f"The given path doesn't exist: {model_path}.")

        arch_path = os.path.join(model_path, "arch.yaml")
        weight_path = os.path.join(model_path, "weights.h5")
        conf_path = os.path.join(model_path, "configurations.yaml")

        return arch_path, weight_path, conf_path

    def _get_model_from_yaml(self, arch_path, custom_objects=None):
        return model_from_yaml(open(arch_path, "r").read(), custom_objects=custom_objects)

