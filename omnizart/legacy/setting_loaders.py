# pylint: disable=R0903,C0115

from omnizart import SETTING_DIR
from omnizart.utils import json_serializable


@json_serializable(key_path="./General", value_path="./Value")
class ChordSettingsVamp(Settings):
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

