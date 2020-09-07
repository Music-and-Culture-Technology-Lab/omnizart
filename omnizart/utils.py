"""Various utility functions for this project."""
# pylint: disable=W0212,R0915
import os
import re
import pickle
import logging
import uuid

import yaml
import librosa
import jsonschema


def get_logger(name=None, level="info"):
    logger_name = str(uuid.uuid4())[:8] if name is None else name
    logger = logging.getLogger(logger_name)

    msg_format = "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]"
    date_format = "%Y-%m-%d %H:%M:%S"
    if len(logger.handlers) == 0:
        formatter = logging.Formatter(fmt=msg_format, datefmt=date_format)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    level_mapping = {
        "info": logging.INFO,
        "warn": logging.WARN,
        "warning": logging.WARNING,
        "debug": logging.DEBUG,
        "error": logging.ERROR
    }
    level = os.environ.get("LOG_LEVEL", level)
    logger.setLevel(level_mapping[level.lower()])
    return logger


def dump_pickle(data, save_to):
    """Dump data to the given path.

    Parameters
    ----------
    data: python objects
        Data to store. Should be python built-in types like `dict`, `list`, `str`, `int`, etc
    save_to: Path
        The full path to store the pickle file, including file name.
        Will create the directory if the given path doesn't exist.

    """
    base_dir = os.path.dirname(save_to)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    with open(save_to, "wb") as pkl_file:
        pickle.dump(data, pkl_file)


def load_pickle(pickle_file):
    """Load pickle file from the given path

    Read and returns the data from the given pickle file path

    Parameters
    ----------
    pickle_file: Path
        The full path to the pickle file for read

    Returns
    -------
    object
        Python object, could be `dict`, `list`, `str`, etc.
    """
    return pickle.load(open(pickle_file, "rb"))


def load_audio_with_librosa(audio_path, sampling_rate=44100):
    """Load audio from the given path with librosa.load

    Parameters
    ----------
    audio_path: Path
        Path to the audio.
    sampling_rate: int
        Target sampling rate after loaded.

    Returns
    -------
    audio: 1D numpy array
        Raw data of the audio.
    tar_sampling_rate: int
        Sampling rate of the audio. Will be the same as the given ``sampling_rate``.
    """
    return librosa.load(audio_path, mono=True, sr=sampling_rate)


def load_yaml(yaml_path):
    return yaml.load(open(yaml_path, "r"), Loader=yaml.Loader)


def write_yaml(json_obj, output_path):
    open(output_path, "w").write(yaml.dump(json_obj))


def camel_to_snake(string):
    """Convert a camel case to snake case"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", string).lower()


def snake_to_camel(string):
    """Convert a snake case to camel case"""
    return "".join(word.title() for word in string.split("_"))


def json_serializable(key_path="./", value_path="./"):
    """Class-level decorator for making a class json-serializable object.

    This decorator makes a class serializable as a json object. All attributes
    will be outputed as a key-value pair recursively if there are also attributes
    that is type of json-serializable class.
    Inject two functions to the decorated class: from_json, to_json.

    Parameters
    ----------
    key_path: Path
        Access sub-object according to the path. E.g. Assume you have a
        dictionary: d = {"a": {"b": {"c": "d"}}}, and the path: p = "a/b",
        then the sub-object after being propagated would be {"c": "d"}.
    value_path: Path
        The relative path to the key_path. This parameter makes you able
        to access the value that is not at the same level as the key.
        E.g. assume you have a sub-object after the propagation of key_path:
        d = {"a": {"b": {"c": "d"}}}, and the value_path: vp = "a/b/c",
        the corresponding value of the key should be "d".

    Examples
    --------
    .. code-block:: python

        >>> @json_serializable()
            class A:
                def __init__(self):
                    self.a = 1
                    self.b = "Hello"
                    self.c = [10, 20, 30]

            @json_serializable()
            class B:
                def __init__(self):
                    self.a_instance = A()
                    self.d = "World"

        >>> inst = B()
        >>> inst.to_json()
        {
            "d": "World",
            "a_instance": {
                "a": 1,
                "b": "Hello",
                "c": [10, 20, 30]
            }
        }

    Notes
    -----
    The attributes should be defined inside '__init__', or those defined as
    class attributes will not be serialized.

    """
    def from_json(self, json_obj):
        if self.schema is not None:
            jsonschema.validate(instance=json_obj, schema=self.schema)
        if self.key_path.startswith("./"):
            self.key_path = self.key_path[2:]
        if self.value_path.startswith("./"):
            self.value_path = self.value_path[2:]

        # Get sub-object containing the target key-value pairs accroding to key_path.
        k_obj = json_obj
        if len(self.key_path) != 0:
            for key in self.key_path.split("/"):
                k_obj = k_obj[key]

        # Iterate through the target key-value pair, and assign the value to the object
        for key in self.__dict__:
            if key in self._ignore_list:
                continue

            camel_key = snake_to_camel(key)
            if hasattr(self.__dict__[key], "from_json"):
                # Another json-seriallizable object
                self.__dict__[key].from_json(k_obj[camel_key])
            elif camel_key not in k_obj:
                raise AttributeError(
                    f"Attribute '{camel_key}' is not defined in configuration file for class {type(self)}!"
                )
            else:
                # Parse the value according to value_path.
                # The path is relative to key_path.
                value = k_obj[camel_key]
                if len(self.value_path) != 0:
                    for v_key in self.value_path.split("/"):
                        value = value[v_key]
                self.__dict__[key] = value

    def to_json(self):
        if self.key_path.startswith("./"):
            self.key_path = self.key_path[2:]
        if self.value_path.startswith("./"):
            self.value_path = self.value_path[2:]

        json_obj = {}
        ref = json_obj
        if len(self.key_path) != 0:
            for key in self.key_path.split("/"):
                camel_key = snake_to_camel(key)
                ref[camel_key] = {}
                ref = ref[camel_key]

        for key, value in self.__dict__.items():
            if key in self._ignore_list:
                continue

            camel_key = snake_to_camel(key)
            if hasattr(value, "to_json"):
                ref[camel_key] = value.to_json()
            elif len(self.value_path) != 0:
                ref[camel_key] = {}
                v_ref = ref[camel_key]
                v_keys = self.value_path.split("/")
                for v_key in v_keys[:-1]:
                    camel_v_key = snake_to_camel(v_key)
                    v_ref[camel_v_key] = {}
                    v_ref = v_ref[camel_v_key]
                v_ref[v_keys[-1]] = value
            else:
                ref[camel_key] = value
        return json_obj

    def wrapper(tar_cls):
        setattr(tar_cls, "from_json", from_json)
        setattr(tar_cls, "to_json", to_json)
        setattr(tar_cls, "key_path", key_path)
        setattr(tar_cls, "value_path", value_path)
        setattr(tar_cls, "schema", None)
        setattr(tar_cls, "_ignore_list", ["_ignore_list", "key_path", "value_path", "schema"])
        return tar_cls

    return wrapper
