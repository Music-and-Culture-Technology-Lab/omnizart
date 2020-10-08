# pylint: disable=C0103,W0612,E0611
import os
from os.path import join as jpath
import time

import h5py

from omnizart.feature.wrapper_func import extract_patch_cqt
from omnizart.drum.prediction import predict
from omnizart.drum.labels import extract_label_13_inst
from omnizart.models.spectral_norm_net import ConvSN2D
from omnizart.utils import get_logger, ensure_path_exists, parallel_generator
from omnizart.base import BaseTranscription
from omnizart.setting_loaders import DrumSettings
from omnizart.constants.datasets import PopStructure


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
        logger.info("Extracting feature...")
        patch_cqt_feature, _ = extract_patch_cqt(input_audio)

        # Load model configurations
        logger.info("Loading model...")
        model, model_settings = self._load_model(model_path, custom_objects=self.custom_objects)

        logger.info("Predicting...")
        pred = predict(patch_cqt_feature, model, model_settings.feature.mini_beat_per_segment)
        logger.debug("Prediction shape: %s", pred.shape)
        return pred

    def generate_feature(self, dataset_path, drum_settings=None):
        if drum_settings is not None:
            assert isinstance(drum_settings, DrumSettings)
            settings = drum_settings
        else:
            settings = self.settings

        # Resolve feature output path
        if settings.dataset.feature_save_path == "+":
            base_output_path = dataset_path
            settings.dataset.save_path = dataset_path
        else:
            base_output_path = settings.dataset.feature_save_path
        train_feat_out_path = jpath(base_output_path, "train_feature")
        test_feat_out_path = jpath(base_output_path, "test_feature")
        ensure_path_exists(train_feat_out_path)
        ensure_path_exists(test_feat_out_path)
        logger.info("Output training feature to %s", train_feat_out_path)
        logger.info("Output testing feature to %s", test_feat_out_path)

        struct = PopStructure()
        train_wavs = struct.get_train_wavs(dataset_path=dataset_path)
        train_labels = struct.get_train_labels(dataset_path=dataset_path)
        logger.info(
            "Start extract training feature of the dataset. "
            "This may take time to finish and affect the computer's performance"
        )
        assert len(train_wavs) == len(train_labels)
        _parallel_feature_extraction(train_wavs, train_labels, train_feat_out_path, settings.feature)

        test_wavs = struct.get_test_wavs(dataset_path=dataset_path)
        test_labels = struct.get_test_labels(dataset_path=dataset_path)
        logger.info(
            "Start extract testing feature of the dataset. "
            "This may take time to finish and affect the computer's performance"
        )
        assert len(test_wavs) == len(test_labels)
        _parallel_feature_extraction(test_wavs, test_labels, test_feat_out_path, settings.feature)


def _parallel_feature_extraction(wav_paths, label_paths, out_path, feat_settings):
    label_path_mapping = _gen_wav_label_path_mapping(label_paths)
    iters = enumerate(
        parallel_generator(
            _all_in_one_extract,
            wav_paths,
            max_workers=2,
            use_thread=True,
            chunk_size=2,
            label_path_mapping=label_path_mapping,
            feat_settings=feat_settings
        )
    )
    for idx, ((patch_cqt, m_beat_arr, label_128, label_13), audio_idx) in iters:
        audio = wav_paths[audio_idx]
        print(f"Progress: {idx+1}/{len(wav_paths)} - {audio}" + " "*6, end="\r")  # noqa: E226

        basename = os.path.basename(audio)
        filename, _ = os.path.splitext(basename)
        out_hdf = jpath(out_path, filename + ".hdf")

        saved = False
        retry_times = 5
        for retry in range(retry_times):
            if saved:
                break
            try:
                with h5py.File(out_hdf, "w") as out_f:
                    out_f.create_dataset("feature", data=patch_cqt, compression="gzip", compression_opts=3)
                    out_f.create_dataset("label", data=label_13, compression="gzip", compression_opts=3)
                    out_f.create_dataset("label_128", data=label_128, compression="gzip", compression_opts=3)
                    out_f.create_dataset("cqt_mini_beat_arr", data=m_beat_arr, compression="gzip", compression_opts=3)
            except OSError as exp:
                logger.warning("OSError occurred, retrying %d times. Reason: %s", retry + 1, str(exp))
                time.sleep(0.5 * 2**retry)
        if not saved:
            logger.error("H5py failed to save the feature file after %d retries.", retry_times)
            raise OSError
    print("")


def _all_in_one_extract(wav_path, label_path_mapping, feat_settings):
    patch_cqt, m_beat_arr = extract_patch_cqt(
        wav_path, sampling_rate=feat_settings.sampling_rate, hop_size=feat_settings.hop_size
    )

    label_path = label_path_mapping[os.path.basename(wav_path)]
    label_128, label_13 = extract_label_13_inst(
        label_path, m_beat_arr, sampling_rate=feat_settings.sampling_rate
    )
    return patch_cqt, m_beat_arr, label_128, label_13


def _gen_wav_label_path_mapping(label_paths):
    mapping = {}
    for label_path in label_paths:
        f_name = os.path.basename(label_path).replace(".mid", ".wav")
        wav_name = f_name.replace("align_mid", "ytd_audio")
        mapping[wav_name] = label_path
    return mapping


if __name__ == "__main__":
    audio_path = "checkpoints/ytd_audio_00105_TRFSJUR12903CB23E7.mp3.wav"
    audio_path = "checkpoints/ytd_audio_00088_TRBHGWP128E0793AD8.mp3.wav"
    app = DrumTranscription()
    pred = app.transcribe(audio_path)
