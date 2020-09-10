import os
from abc import ABCMeta, abstractmethod

from tensorflow.keras.models import model_from_yaml

from omnizart import MODULE_PATH, SETTING_DIR
from omnizart.utils import load_yaml, get_logger


logger = get_logger("Base Class")


class BaseTranscription(metaclass=ABCMeta):
    """Base class of transcription applications."""
    def __init__(self, setting_class):
        self.setting_class = setting_class

        default_conf_path = os.path.join(SETTING_DIR, setting_class.default_setting_file)
        logger.debug("Loading default configurations: %s", default_conf_path)
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
        elif not os.path.basename(model_path).startswith(self.settings.model.save_prefix.lower()) \
                and not set(["arch.yaml", "weights.h5", "configurations.yaml"]).issubset(os.listdir(model_path)):
            # Search checkpoint folders under the given path
            dirs = [c_dir for c_dir in os.listdir(model_path) if os.path.isdir(c_dir)]
            prefix = self.settings.model.save_prefix.lower()
            cand_dirs = [c_dir for c_dir in dirs if c_dir.startswith(prefix)]

            if len(cand_dirs) == 0:
                raise FileNotFoundError(f"No checkpoint of {prefix} found in {model_path}.")
            elif len(cand_dirs) > 1:
                logger.warning("There are multiple checkpoints in the directory. Default to use %s", cand_dirs[0])
            model_path = os.path.join(model_path, cand_dirs[0])

        arch_path = os.path.join(model_path, "arch.yaml")
        weight_path = os.path.join(model_path, "weights.h5")
        conf_path = os.path.join(model_path, "configurations.yaml")

        return arch_path, weight_path, conf_path

    def _get_model_from_yaml(self, arch_path, custom_objects=None):
        return model_from_yaml(open(arch_path, "r").read(), custom_objects=custom_objects)
