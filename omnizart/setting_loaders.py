"""Define classes for loading YAML setting files.

Parse settings into pre-defined classes, similar to the 'view model'
concept in MVVC, and instead of access values key by key.
"""

from omnizart.utils import json_serializable
from omnizart.constants.schema.music_settings import MUSIC_SETTINGS_SCHEMA


@json_serializable(key_path="./General", value_path="./Value")
class MusicSettings:
    def __init__(self):
        self.feature = self.MusicFeature()
        self.dataset = self.MusicDataset()
        self.model = self.MusicModel()
        self.inference = self.MusicInference()
        self.training = self.MusicTraining()
        self.transcription_mode: str = None

        # As a json-serializable object, if variable 'schema' is set,
        # then the input json object will be validated when parsing
        # settings using the from_json function.
        self.schema = MUSIC_SETTINGS_SCHEMA

    @json_serializable(key_path="./Settings", value_path="./Value")
    class MusicInference:
        def __init__(self):
            self.min_length: float = None
            self.inst_th: float = None
            self.onset_th: float = None
            self.dura_th: float = None
            self.frame_th: float = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class MusicFeature:
        def __init__(self):
            self.hop_size: float = None
            self.sampling_rate: int = None
            self.window_size: int = None
            self.frequency_resolution: float = None
            self.frequency_center: float = None
            self.time_center: float = None
            self.gamma: list = None
            self.bins_per_octave: int = None
            self.harmonic_number: int = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class MusicDataset:
        def __init__(self):
            self.save_path: str = None
            self.feature_type: str = None
            self.feature_save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class MusicModel:
        def __init__(self):
            self.checkpoint_path: str = None
            self.save_prefix: str = None
            self.save_path: str = None
    
    @json_serializable(key_path="./Settings", value_path="./Value")
    class MusicTraining:
        def __init__(self):
            self.epoch: int = None
            self.steps: int = None
            self.val_steps: int = None
            self.batch_size: int = None
            self.val_batch_size: int = None
            self.early_stop: int = None
            self.loss_function: str = None
            self.label_type: str = None
            self.channels: list = None
            self.harmonic: bool = None
            self.timesteps: int = None


from omnizart.utils import load_yaml
if __name__ == "__main__":
    obj = load_yaml("omnizart/defaults/music.yaml")
    inst = MusicSettings()
    inst.from_json(obj)
