"""Define classes for loading YAML setting files.

Parse settings into pre-defined classes, similar to the 'view model'
concept in MVVC, and instead of access values key by key.
"""
# pylint: disable=R0903,C0115
import os

from omnizart import SETTING_DIR
from omnizart.io import load_yaml
from omnizart.utils import json_serializable
from omnizart.constants.schema.music_settings import MUSIC_SETTINGS_SCHEMA


class Settings:
    default_setting_file = None

    def __init__(self, conf_path=None):
        # Load default settings
        if conf_path is not None:
            self.from_json(load_yaml(conf_path))  # pylint: disable=E1101
        else:
            conf_path = os.path.join(SETTING_DIR, self.default_setting_file)
            self.from_json(load_yaml(conf_path))  # pylint: disable=E1101


@json_serializable(key_path="./General", value_path="./Value")
class MusicSettings(Settings):
    """Hello"""
    default_setting_file: str = "music.yaml"

    def __init__(self, conf_path=None):
        self.feature = self.MusicFeature()
        self.dataset = self.MusicDataset()
        self.model = self.MusicModel()
        self.inference = self.MusicInference()
        self.training = self.MusicTraining()
        self.transcription_mode: str = None
        self.checkpoint_path: str = None

        # As a json-serializable object, if variable 'schema' is set,
        # then the input json object will be validated when parsing
        # settings using the from_json function.
        self.schema = MUSIC_SETTINGS_SCHEMA

        super().__init__(conf_path=conf_path)

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
            self.harmonic: bool = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class MusicDataset:
        def __init__(self):
            self.save_path: str = None
            self.feature_type: str = None
            self.feature_save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class MusicModel:
        def __init__(self):
            self.save_prefix: str = None
            self.save_path: str = None
            self.model_type: str = None

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
            self.timesteps: int = None
            self.feature_num: int = None


@json_serializable(key_path="./General", value_path="./Value")
class DrumSettings(Settings):
    default_setting_file: str = "drum.yaml"

    def __init__(self, conf_path=None):
        self.transcription_mode: str = None
        self.checkpoint_path: dict = None
        self.feature = self.DrumFeature()
        self.dataset = self.DrumDataset()
        self.model = self.DrumModel()
        self.inference = self.DrumInference()
        self.training = self.DrumTraining()

        super().__init__(conf_path=conf_path)

    @json_serializable(key_path="./Settings", value_path="./Value")
    class DrumFeature:
        def __init__(self):
            self.sampling_rate: int = None
            self.padding_seconds: float = None
            self.lowest_note: int = None
            self.number_of_notes: int = None
            self.hop_size: int = None
            self.mini_beat_per_bar: int = None
            self.mini_beat_per_segment: int = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class DrumDataset:
        def __init__(self):
            self.save_path: str = None
            self.feature_save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class DrumModel:
        def __init__(self):
            self.save_prefix: str = None
            self.save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class DrumInference:
        def __init__(self):
            self.bass_drum_th: float = None
            self.snare_th: float = None
            self.hihat_th: float = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class DrumTraining:
        def __init__(self):
            self.epoch: int = None
            self.steps: int = None
            self.val_steps: int = None
            self.batch_size: int = None
            self.val_batch_size: int = None
            self.early_stop: int = None
            self.init_learning_rate: float = None
            self.res_block_num: int = None


@json_serializable(key_path="./General", value_path="./Value")
class ChordSettings(Settings):
    default_setting_file: str = "chord.yaml"

    def __init__(self, conf_path=None):
        self.transcription_mode: str = None
        self.checkpoint_path: dict = None
        self.feature = self.ChordFeature()
        self.dataset = self.ChordDataset()
        self.model = self.ChordModel()
        self.inference = self.ChordInference()
        self.training = self.ChordTraining()

        super().__init__(conf_path=conf_path)

    @json_serializable(key_path="./Settings", value_path="./Value")
    class ChordFeature:
        def __init__(self):
            self.segment_width: int = None
            self.segment_hop: int = None
            self.num_steps: int = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class ChordDataset:
        def __init__(self):
            self.save_path: str = None
            self.feature_save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class ChordModel:
        def __init__(self):
            self.save_prefix: str = None
            self.save_path: str = None
            self.num_enc_attn_blocks: int = None
            self.num_dec_attn_blocks: int = None
            self.freq_size: int = None
            self.enc_input_emb_size: int = None
            self.dec_input_emb_size: int = None
            self.dropout_rate: float = None
            self.annealing_rate: float = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class ChordInference:
        def __init__(self):
            self.min_dura: float = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class ChordTraining:
        def __init__(self):
            self.epoch: int = None
            self.steps: int = None
            self.val_steps: int = None
            self.batch_size: int = None
            self.val_batch_size: int = None
            self.early_stop: int = None
            self.init_learning_rate: float = None
            self.learning_rate_decay: float = None


@json_serializable(key_path="./General", value_path="./Value")
class VocalContourSettings(Settings):
    default_setting_file: str = "vocal_contour.yaml"

    def __init__(self, conf_path=None):
        self.transcription_mode: str = None
        self.checkpoint_path: str = None
        self.feature = self.VocalContourFeature()
        self.dataset = self.VocalContourDataset()
        self.model = self.VocalContourModel()
        self.training = self.VocalContourTraining()

        super().__init__(conf_path=conf_path)

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalContourFeature():
        def __init__(self):
            self.hop_size: float = None
            self.sampling_rate: int = None
            self.window_size: int = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalContourDataset():
        def __init__(self):
            self.save_path: str = None
            self.feature_save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalContourModel():
        def __init__(self):
            self.save_prefix: str = None
            self.save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalContourTraining():
        def __init__(self):
            self.epoch: int = None
            self.early_stop: int = None
            self.steps: int = None
            self.val_steps: int = None
            self.batch_size: int = None
            self.val_batch_size: int = None
            self.timesteps: int = None


@json_serializable(key_path="./General", value_path="./Value")
class VocalSettings(Settings):
    default_setting_file: str = "vocal.yaml"

    def __init__(self, conf_path=None):
        self.transcription_mode: str = None
        self.checkpoint_path: dict = None
        self.feature = self.VocalFeature()
        self.dataset = self.VocalDataset()
        self.model = self.VocalModel()
        self.inference = self.VocalInference()
        self.training = self.VocalTraining()

        super().__init__(conf_path=conf_path)

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalFeature:
        def __init__(self):
            self.hop_size: float = None
            self.sampling_rate: int = None
            self.frequency_resolution: float = None
            self.frequency_center: float = None
            self.time_center: float = None
            self.gamma: list = None
            self.bins_per_octave: int = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalDataset:
        def __init__(self):
            self.save_path: str = None
            self.feature_save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalModel:
        def __init__(self):
            self.save_prefix: str = None
            self.save_path: str = None
            self.min_kernel_size: int = None
            self.depth: int = None
            self.shake_drop: bool = True
            self.alpha: int = None
            self.semi_loss_weight: float = None
            self.semi_xi: float = None
            self.semi_epsilon: float = None
            self.semi_iterations: int = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalInference:
        def __init__(self):
            self.context_length: int = None
            self.threshold: float = None
            self.min_duration: float = None
            self.pitch_model: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalTraining:
        def __init__(self):
            self.epoch: int = None
            self.steps: int = None
            self.val_steps: int = None
            self.batch_size: int = None
            self.val_batch_size: int = None
            self.early_stop: int = None
            self.init_learning_rate: float = None
            self.context_length: int = None


@json_serializable(key_path="./General", value_path="./Value")
class PatchCNNSettings(Settings):
    default_setting_file: str = "patch_cnn.yaml"

    def __init__(self, conf_path=None):
        self.transcription_mode: str = None
        self.checkpoint_path: dict = None
        self.feature = self.PatchCNNFeature()
        self.model = self.PatchCNNModel()
        self.dataset = self.PatchCNNDataset()
        self.training = self.PatchCNNTraining()
        self.inference = self.PatchCNNInference()

        super().__init__(conf_path=conf_path)

    @json_serializable(key_path="./Settings", value_path="./Value")
    class PatchCNNFeature:
        def __init__(self):
            self.patch_size: int = None
            self.peak_threshold: float = None
            self.sampling_rate: int = None
            self.hop_size: float = None
            self.window_size: int = None
            self.frequency_resolution: float = None
            self.frequency_center: float = None
            self.time_center: float = None
            self.gamma: list = None
            self.bins_per_octave: int = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class PatchCNNModel:
        def __init__(self):
            self.save_prefix: str = None
            self.save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class PatchCNNDataset:
        def __init__(self):
            self.save_path: str = None
            self.feature_save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class PatchCNNTraining:
        def __init__(self):
            self.epoch: int = None
            self.steps: int = None
            self.val_steps: int = None
            self.batch_size: int = None
            self.val_batch_size: int = None
            self.early_stop: int = None
            self.init_learning_rate: float = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class PatchCNNInference:
        def __init__(self):
            self.threshold: float = None
            self.max_method: str = None


@json_serializable(key_path="./General", value_path="./Value")
class BeatSettings(Settings):
    default_setting_file: str = "beat.yaml"

    def __init__(self, conf_path=None):
        self.transcription_mode: str = None
        self.checkpoint_path: dict = None
        self.feature = self.BeatFeature()
        self.dataset = self.BeatDataset()
        self.model = self.BeatModel()
        self.training = self.BeatTraining()
        self.inference = self.BeatInference()

        super().__init__(conf_path=conf_path)

    @json_serializable(key_path="./Settings", value_path="./Value")
    class BeatFeature:
        def __init__(self):
            self.time_unit: float = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class BeatDataset:
        def __init__(self):
            self.save_path: str = None
            self.feature_save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class BeatModel:
        def __init__(self):
            self.save_prefix: str = None
            self.save_path: str = None
            self.model_type: str = None
            self.timesteps: int = None
            self.lstm_hidden_dim: int = None
            self.num_lstm_layers: int = None
            self.attn_hidden_dim: int = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class BeatTraining:
        def __init__(self):
            self.epoch: int = None
            self.steps: int = None
            self.val_steps: int = None
            self.batch_size: int = None
            self.val_batch_size: int = None
            self.early_stop: int = None
            self.init_learning_rate: float = None
            self.down_beat_weight: float = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class BeatInference:
        def __init__(self):
            self.beat_threshold: float = None
            self.down_beat_threshold: float = None
            self.min_distance: float = None
