# pylint: disable=C0103,W0612

import os

import numpy as np

from omnizart.feature.cfp import extract_cfp
from omnizart.feature.hcfp import extract_hcfp
from omnizart.music.inference import multi_inst_note_inference
from omnizart.music.model_manager import ModelManager
from omnizart.base import BaseTranscription
from omnizart.io_utils import load_yaml
from omnizart.setting_loaders import MusicSettings
from omnizart.constants.midi import MUSICNET_INSTRUMENT_PROGRAMS


class MusicTranscription(BaseTranscription):
    """Application class for music transcription.

    Inherited from the BaseTranscription class to make sure everything
    needed got override.
    """
    def __init__(self, conf_path=None):
        super().__init__()
        
        # Load default settings from yaml file.
        default_conf_path = os.path.join(self.conf_dir, "music.yaml")
        json_obj = load_yaml(default_conf_path)
        if conf_path is not None:
            override_json_obj = load_yaml(conf_path)
        json_obj.update(override_json_obj)
        self.settings = MusicSettings().from_json(json_obj)

        self.m_manage = ModelManager()

    def transcribe(input_audio, model_path=None, output="./"):
        """Transcribe notes and instruments of the given audio.

        This function transcribes notes (onset, duration) of each instruments in the audio.
        The results will be written out as a MIDI file.

        Parameters
        ----------
        input_audio: Path
            Path to the wav audio file.
        model_path: Path
            Path to the trained model. Should be the folder that contains `arch.yaml`, `weights.h5`, and
            `configuration.csv`.
        output: Path (optional)
            Path for writing out the transcribed MIDI file. Default to current path.

        See Also
        --------
        omnizart.cli.music.transcribe: The coressponding command line entry.
        """
        if not os.path.isfile(input_audio):
            raise FileNotFoundError(f"The given audio path does not exist. Path: {input_audio}")

        if model_path is None:
            model_path = self.settings.Model.CheckpointPath[self.settings.TranscriptionMode]

        model = self.m_manage.load_model(model_path)
        print(self.m_manage)

        # TODO: Add feature-related settings to the configuration.json and load it in ModelManager
        print("Extracting feature...")
        if self.m_manage.feature_type == "HCFP":
            spec, gcos, ceps, cenf = extract_hcfp(input_audio)
            feature = np.dstack([spec, gcos, ceps])
        else:
            z, spec, gcos, ceps, cenf = extract_cfp(input_audio)
            feature = np.dstack([z.T, spec.T, gcos.T, ceps.T])

        mode_mapping = {
            "frame": "true_frame",
            "frame_onset": "note",
            "multi_instrument_frame": "true_frame",
            "multi_instrument_note": "note-stream",
        }

        print("Predicting...")
        pred = self.m_manage.predict(feature[:, :, self.m_manage.input_channels], model)

        print("Infering notes....")
        midi = multi_inst_note_inference(
            pred,
            mode=mode_mapping[m_manage.label_type],
            onset_th=self.m_manage.onset_th,
            dura_th=self.m_manage.dura_th,
            frm_th=self.m_manage.frm_th,
            inst_th=self.m_manage.inst_th,
            t_unit=self.settings.Feature.HopSize,
            channel_program_mapping=MUSICNET_INSTRUMENT_PROGRAMS,
        )

        save_to = os.path.join(output, os.path.basename(input_audio).replace(".wav", ".mid"))
        midi.write(save_to)
        print(f"MIDI file has been written to {save_to}")

