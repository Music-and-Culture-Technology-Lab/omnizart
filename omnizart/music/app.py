"""Application class of music.

Inludes core functions and interfaces for transcribing the audio, train
a model, generate feature of datasets, and evaluate on models.

See Also
--------
omnizart.base.BaseTranscription: The base class of all transcription/application classes.
"""

# pylint: disable=C0103,W0612,E0611
import os

import numpy as np
from scipy.special import expit

from omnizart.feature.cfp import extract_cfp
from omnizart.feature.hcfp import extract_hcfp
from omnizart.music.inference import multi_inst_note_inference
from omnizart.music.utils import create_batches, cut_batch_pred, cut_frame
from omnizart.base import BaseTranscription
from omnizart.utils import get_logger
from omnizart.setting_loaders import MusicSettings
from omnizart.constants.midi import MUSICNET_INSTRUMENT_PROGRAMS
from omnizart.constants.feature import FEATURE_NAME_TO_NUMBER

logger = get_logger("Music Transcription")


class MusicTranscription(BaseTranscription):
    """Application class for music transcription.

    Inherited from the BaseTranscription class to make sure everything
    needed got override.
    """
    def __init__(self):
        super().__init__(MusicSettings)

    def transcribe(self, input_audio, model_path=None, output="./"):
        """Transcribe notes and instruments of the given audio.

        This function transcribes notes (onset, duration) of each instruments in the audio.
        The results will be written out as a MIDI file.

        Parameters
        ----------
        input_audio: Path
            Path to the wav audio file.
        model_path: Path
            Path to the trained model. Should be the folder that contains `arch.yaml`, `weights.h5`, and
            `configuration.yaml`.
        output: Path (optional)
            Path for writing out the transcribed MIDI file. Default to current path.

        See Also
        --------
        omnizart.cli.music.transcribe: The coressponding command line entry.
        """
        if not os.path.isfile(input_audio):
            raise FileNotFoundError(f"The given audio path does not exist. Path: {input_audio}")

        logger.info("Loading model...")
        model, model_settings = self._load_model(model_path)

        logger.info("Extracting feature...")
        if model_settings.feature.harmonic:
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
            "note": "note",
            "note-stream": "note-stream",
            "frame-stream": "true_frame",
        }

        logger.info("Predicting...")
        channels = [FEATURE_NAME_TO_NUMBER[ch_name] for ch_name in model_settings.training.channels]
        pred = self._predict(feature[:, :, channels], model, timesteps=model_settings.training.timesteps)

        logger.info("Infering notes....")
        midi = multi_inst_note_inference(
            pred,
            mode=mode_mapping[model_settings.training.label_type],
            onset_th=model_settings.inference.onset_th,
            dura_th=model_settings.inference.dura_th,
            frm_th=model_settings.inference.frame_th,
            inst_th=model_settings.inference.inst_th,
            t_unit=model_settings.feature.hop_size,
            channel_program_mapping=MUSICNET_INSTRUMENT_PROGRAMS,
        )

        if output is not None:
            save_to = os.path.join(output, os.path.basename(input_audio).replace(".wav", ".mid"))
            midi.write(save_to)
            logger.info("MIDI file has been written to %s", save_to)
        return midi

    def _predict(self, feature, model, timesteps=128, feature_num=384, batch_size=4):
        """Make predictions on the feature.

        Generate predictions by using the loaded model.

        Parameters
        ----------
        feature: numpy.ndarray
            Extracted feature of the audio.
            Dimension:  timesteps x feature_size x channels
        model: keras.Model
            The loaded model instance
        feature_num: int
            Padding along the feature dimension to the size `feature_num`
        batch_size: int
            Batch size for each step of prediction. The size is depending on the available GPU memory.

        Returns
        -------
        pred: numpy.ndarray
            The predicted results. The values are ranging from 0~1.
        """

        # Create batches of the feature
        features = create_batches(feature, b_size=batch_size, timesteps=timesteps, feature_num=feature_num)

        # Container for the batch prediction
        pred = []

        # Initiate lamda function for later processing of prediction
        cut_frm = lambda x: cut_frame(x, ori_feature_size=352, feature_num=features[0][0].shape[1])

        t_len = len(features[0][0])
        first_split_start = round(t_len * 0.75)
        second_split_start = t_len + round(t_len * 0.25)

        total_batches = len(features)
        features.insert(0, [np.zeros_like(features[0][0])])
        features.append([np.zeros_like(features[0][0])])
        logger.debug("Total batches: %d", total_batches)
        for i in range(1, total_batches + 1):
            print("batch: {}/{}".format(i, total_batches), end="\r")
            first_half_batch = []
            second_half_batch = []
            b_size = len(features[i])
            features[i] = np.insert(features[i], 0, features[i - 1][-1], axis=0)
            features[i] = np.insert(features[i], len(features[i]), features[i + 1][0], axis=0)
            for ii in range(1, b_size + 1):
                ctx = np.concatenate(features[i][ii - 1:ii + 2], axis=0)

                first_half = ctx[first_split_start:first_split_start + t_len]
                first_half_batch.append(first_half)

                second_half = ctx[second_split_start:second_split_start + t_len]
                second_half_batch.append(second_half)

            p_one = model.predict(np.array(first_half_batch), batch_size=b_size)
            p_two = model.predict(np.array(second_half_batch), batch_size=b_size)
            p_one = cut_batch_pred(p_one)
            p_two = cut_batch_pred(p_two)

            for ii in range(b_size):
                frm = np.concatenate([p_one[ii], p_two[ii]])
                pred.append(cut_frm(frm))

        pred = expit(np.concatenate(pred))  # sigmoid function, mapping the ReLU output value to [0, 1]
        return pred
