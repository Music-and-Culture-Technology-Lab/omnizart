# pylint: disable=C0103,W0612,E0611
import os

import numpy as np

from omnizart.feature.cqt import extract_cqt
from omnizart.feature.beat_for_drum import extract_mini_beat_from_audio_path
from omnizart.drum.utils import extract_patch_cqt, create_batches, merge_batches
from omnizart.models.spectral_norm_net import ConvSN2D
from omnizart.utils import get_logger
from omnizart.base import BaseTranscription
from omnizart.setting_loaders import DrumSettings


logger = get_logger("Drum Transcription")


class DrumTranscription(BaseTranscription):
    """Application class for drum transcriptions."""
    def __init__(self):
        super().__init__(DrumSettings)
        self.custom_objects = {"ConvSN2D": ConvSN2D}

    def transcribe(self, input_audio, model_path=None, output="./"):
        """Transcribe drum in the audio.

        This function transcribes drum activations in the music. Currently the model
        predicts 13 classes of different drum sets, and 3 of them will be written to
        the MIDI file.

        Parameters
        ----------
        input_audio: Path
            Path to the raw audio file (.wav).
        model_path: Path
            Path to the trained model.
        output: Path (optional)
            Path for writing out the transcribed MIDI file. Default to current path.

        See Also
        --------
        omnizart.cli.drum.transcribe: CLI entry point of this function.
        """
        if not os.path.isfile(input_audio):
            raise FileNotFoundError(f"The given audio path does not exist. Path: {input_audio}")

        # Extract feature according to model configuration
        logger.info("Extracting CQT feature...")
        cqt_feature = extract_cqt(input_audio)

        logger.info("Extracting beat information...")
        mini_beat_arr = extract_mini_beat_from_audio_path(input_audio)

        # Load model configurations
        logger.info("Loading model...")
        model, model_settings = self._load_model(model_path, custom_objects=self.custom_objects)

        logger.debug("CQT feature shape: %s", cqt_feature.shape)
        logger.debug("Mini-beat array shape: %s", mini_beat_arr.shape)

        logger.info("Predicting...")
        pred = self._predict(
            cqt_feature, mini_beat_arr, model, model_settings.feature.mini_beat_per_segment
        )[:len(mini_beat_arr)]  # Remove padding

        logger.debug("Prediction shape: %s", pred.shape)

        return pred

    def _predict(self, cqt_feature, mini_beat_arr, model, mini_beat_per_seg, batch_size=6):
        patch_cqt_feature = extract_patch_cqt(cqt_feature, mini_beat_arr)
        batches = create_batches(patch_cqt_feature, mini_beat_per_seg, b_size=batch_size)
        batch_pred = []
        for idx, batch in enumerate(batches):
            print(f"{idx+1}/{len(batches)}", end="\r")
            batch_pred.append(model.predict(batch))
        return merge_batches(np.array(batch_pred))


if __name__ == "__main__":
    audio_path = "checkpoints/ytd_audio_00105_TRFSJUR12903CB23E7.mp3.wav"
    audio_path = "checkpoints/ytd_audio_00088_TRBHGWP128E0793AD8.mp3.wav"
    app = DrumTranscription()
    pred = app.transcribe(audio_path)
