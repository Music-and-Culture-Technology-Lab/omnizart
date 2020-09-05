import os
import yaml
from abc import ABCMeta, abstractmethod

import omnizart


class BaseTranscription(metaclass=ABCMeta):
    def __init__(self):
        self.conf_dir = os.path.abspath(omnizart.__file__+"/../defaults")

    @abstractmethod
    def transcribe(self, input_audio, model_path, output="./"):
        pass
