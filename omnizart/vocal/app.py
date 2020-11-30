import os
import glob
import shutil
from os.path import join as jpath
from collections import OrderedDict
from datetime import datetime

import h5py
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from spleeter.separator import Separator
from spleeter.utils.logging import get_logger as sp_get_logger

from omnizart.io import load_audio, write_yaml
from omnizart.utils import get_logger, resolve_dataset_type, parallel_generator, ensure_path_exists
from omnizart.constants import datasets as d_struct
from omnizart.base import BaseTranscription, BaseDatasetLoader
from omnizart.feature.cfp import extract_vocal_cfp, _extract_vocal_cfp
from omnizart.setting_loaders import VocalSettings
from omnizart.vocal import labels as lextor
from omnizart.vocal.prediction import predict
from omnizart.train import get_train_val_feat_file_list
from omnizart.models.pyramid_net import PyramidNet


logger = get_logger("Vocal Transcription")


class VocalTranscription(BaseTranscription):
    """Application class for vocal note transcription.

    This application implements the training procedure in a semi-supervised way.
    """
    def __init__(self, conf_path=None):
        super().__init__(VocalSettings, conf_path=conf_path)

        # Disable logging information of Spleeter
        sp_logger = sp_get_logger()
        sp_logger.setLevel(40)  # logging.ERROR

    def transcribe(self, input_audio, model_path=None, output="./"):
        logger.info("Separating vocal track from the audio...")
        separator = Separator('spleeter:2stems')

        # Tricky way to bypass the annoying tensorflow graph was finalized issue.
        separator._params["stft_backend"] = "librosa"  # pylint: disable=protected-access

        wav, fs = load_audio(input_audio, mono=False)
        pred = separator.separate(wav)

        logger.info("Extracting feature...")
        wav = librosa.to_mono(pred["vocals"].squeeze().T)
        feature = _extract_vocal_cfp(wav, fs)

        # Load model configurations
        logger.info("Loading model...")
        model, model_settings = self._load_model(model_path)

        logger.info("Predicting...")
        pred = predict(feature, model)
        return pred, feature

    def generate_feature(self, dataset_path, vocal_settings=None, num_threads=4):
        settings = self._validate_and_get_settings(vocal_settings)

        dataset_type = resolve_dataset_type(
            dataset_path,
            keywords={"cmedia": "cmedia", "mir-1k": "mir1k", "mir1k": "mir1k", "tonas": "tonas"}
        )
        if dataset_type is None:
            logger.warning(
                "The given path %s does not match any built-in processable dataset. Do nothing...",
                dataset_path
            )
            return
        logger.info("Inferred dataset type: %s", dataset_type)

        # Build instance mapping
        struct = {
            "cmedia": d_struct.CMediaStructure,
            "mir1k": d_struct.MIR1KStructure,
            "tonas": d_struct.TonasStructure
        }[dataset_type]
        label_extractor = {
            "cmedia": lextor.CMediaLabelExtraction,
            "mir1k": lextor.MIR1KlabelExtraction,
            "tonas": lextor.TonasLabelExtraction
        }[dataset_type]

        # Fetching wav files
        train_data = struct.get_train_data_pair(dataset_path=dataset_path)
        test_data = struct.get_test_data_pair(dataset_path=dataset_path)
        logger.info("Number of total training wavs: %d", len(train_data))
        logger.info("Number of total testing wavs: %d", len(test_data))

        # Resolve feature output path
        train_feat_out_path, test_feat_out_path = self._resolve_feature_output_path(dataset_path, settings)
        logger.info("Output training feature to %s", train_feat_out_path)
        logger.info("Output testing feature to %s", test_feat_out_path)

        # Feature extraction
        logger.info(
            "Start extract training feature of the dataset %s. "
            "This may take time to finish and affect the computer's performance.",
            dataset_type.title()
        )
        # Do source separation to separate the vocal track first.
        wav_paths = _vocal_separation([data[0] for data in train_data], jpath(dataset_path, "train_wavs_spleeter"))
        train_data = _validate_order_and_get_new_pair(wav_paths, train_data)
        _parallel_feature_extraction(
            train_data, label_extractor, train_feat_out_path, settings.feature, num_threads=num_threads
        )

        # Feature extraction
        logger.info(
            "Start extract testing feature of the dataset %s. "
            "This may take time to finish and affect the computer's performance.",
            dataset_type.title()
        )
        # Do source separation to separate the vocal track first.
        wav_paths = _vocal_separation([data[0] for data in test_data], jpath(dataset_path, "test_wavs_spleeter"))
        test_data = _validate_order_and_get_new_pair(wav_paths, test_data)
        _parallel_feature_extraction(
            test_data, label_extractor, test_feat_out_path, settings.feature, num_threads=num_threads
        )

        # Writing out the settings
        write_yaml(settings.to_json(), jpath(train_feat_out_path, ".success.yaml"))
        write_yaml(settings.to_json(), jpath(test_feat_out_path, ".success.yaml"))
        logger.info("All done")

    def train(
        self, feature_folder, semi_feature_folder=None, model_name=None, input_model_path=None, vocal_settings=None
    ):
        settings = self._validate_and_get_settings(vocal_settings)

        if input_model_path is not None:
            logger.info("Continue to train on model: %s", input_model_path)
            model, prev_set = self._load_model(input_model_path)
            settings.model.save_path = prev_set.model.save_path

        logger.info("Constructing dataset instance")
        split = settings.training.steps / (settings.training.steps + settings.training.val_steps)
        train_feat_files, val_feat_files = get_train_val_feat_file_list(feature_folder, split=split)

        output_types = (tf.float32, tf.float32)
        output_shapes = ((settings.training.context_length*2 + 1, 174, 9), (6))  # noqa: E226
        train_dataset = VocalDatasetLoader(
                ctx_len=settings.training.context_length,
                feature_files=train_feat_files,
                num_samples=settings.training.epoch * settings.training.batch_size * settings.training.steps
            ) \
            .get_dataset(settings.training.batch_size, output_types=output_types, output_shapes=output_shapes)
        val_dataset = VocalDatasetLoader(
                ctx_len=settings.training.context_length,
                feature_files=val_feat_files,
                num_samples=settings.training.epoch * settings.training.val_batch_size * settings.training.val_steps
            ) \
            .get_dataset(settings.training.val_batch_size, output_types=output_types, output_shapes=output_shapes)
        if semi_feature_folder is not None:
            # Semi-supervise learning dataset.
            feat_files = glob.glob(f"{semi_feature_folder}/*.hdf")
            semi_dataset = VocalDatasetLoader(
                    ctx_len=settings.training.context_length,
                    feature_files=feat_files,
                    num_samples=settings.training.epoch * settings.training.batch_size * settings.training.steps
                ) \
                .get_dataset(settings.training.batch_size, output_types=output_types, output_shapes=output_shapes)
            train_dataset = tf.data.Dataset.zip((train_dataset, semi_dataset))

        if input_model_path is None:
            logger.info("Constructing new model")
            model = self.get_model(settings)

        optimizer = tfa.optimizers.AdamW(weight_decay=0.01, learning_rate=settings.training.init_learning_rate)
        model.compile(optimizer=optimizer, loss='bce', metrics=['accuracy', 'binary_accuracy'])

        logger.info("Resolving model output path")
        if model_name is None:
            model_name = str(datetime.now()).replace(" ", "_")
        if not model_name.startswith(settings.model.save_prefix):
            model_name = settings.model.save_prefix + "_" + model_name
        model_save_path = jpath(settings.model.save_path, model_name)
        ensure_path_exists(model_save_path)
        write_yaml(settings.to_json(), jpath(model_save_path, "configurations.yaml"))
        logger.info("Model output to: %s", model_save_path)

        logger.info("Constructing callbacks")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=settings.training.early_stop, monitor="val_loss"),
            tf.keras.callbacks.ModelCheckpoint(
                jpath(model_save_path, "weights"), save_weights_only=True, monitor="val_loss"
            )
        ]
        logger.info("Callback list: %s", callbacks)

        logger.info("Start training")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=settings.training.epoch,
            steps_per_epoch=settings.training.steps,
            validation_steps=settings.training.val_steps,
            callbacks=callbacks,
            use_multiprocessing=True,
            workers=8
        )
        return model_save_path, history

    def get_model(self, settings):
        """Get the Pyramid model.

        More comprehensive reasons to having this method, please refer to
        ``omnizart.base.BaseTranscription.get_model``.
        """
        return PyramidNet(
            out_classes=6,
            min_kernel_size=settings.model.min_kernel_size,
            depth=settings.model.depth,
            alpha=settings.model.alpha,
            shakedrop=settings.model.shake_drop,
            semi_loss_weight=settings.model.semi_loss_weight,
            semi_xi=settings.model.semi_xi,
            semi_epsilon=settings.model.semi_epsilon,
            semi_iters=settings.model.semi_iterations
        )


def _validate_order_and_get_new_pair(wav_paths, data_pair):
    wavs = [os.path.basename(wav) for wav in wav_paths]
    ori_wavs = [os.path.basename(data[0]) for data in data_pair]
    assert wavs == ori_wavs
    return [(wav_path, label_path) for wav_path, (_, label_path) in zip(wav_paths, data_pair)]


def _vocal_separation(wav_list, out_folder):
    wavs = OrderedDict({os.path.basename(wav): wav for wav in wav_list})
    if os.path.exists(out_folder):
        # There are already some separated audio.
        sep_wavs = set(os.listdir(out_folder))
        diff_wavs = set(wavs.keys()) - sep_wavs
        logger.debug("Audio to be separated: %s", diff_wavs)

        # Check the difference of the separated audio and the received audio list.
        done_wavs = set(wavs.keys()) - diff_wavs
        wavs_copy = wavs.copy()
        for dwav in done_wavs:
            del wavs_copy[dwav]
        wav_list = list(wavs_copy.values())

    out_list = [jpath(out_folder, wav) for wav in wavs]
    if len(wav_list) > 0:
        separator = Separator('spleeter:2stems')
        for idx, wav_path in enumerate(wav_list, 1):
            logger.info("Separation Progress: %d/%d - %s", idx, len(wav_list), wav_path)
            separator.separate_to_file(wav_path, out_folder)

            # The separated tracks are stored in sub-folders.
            # Move the vocal track to the desired folder and rename them.
            fname, _ = os.path.splitext(os.path.basename(wav_path))
            sep_folder = jpath(out_folder, fname)
            vocal_track = jpath(sep_folder, "vocals.wav")
            shutil.move(vocal_track, jpath(out_folder, fname + ".wav"))
            shutil.rmtree(sep_folder)
    return out_list


def _all_in_one_extract(data_pair, label_extractor, t_unit, **feat_kargs):
    wav, label = data_pair
    logger.debug("Extracting vocal CFP feature")
    feature = extract_vocal_cfp(wav, **feat_kargs)
    logger.debug("Extracting label")
    label = label_extractor.extract_label(label, t_unit=t_unit)
    return feature, label


def _parallel_feature_extraction(
    data_pair, label_extractor, out_path, feat_settings, num_threads=4
):
    feat_extract_params = {
        "hop": feat_settings.hop_size,
        "fr": feat_settings.frequency_resolution,
        "fc": feat_settings.frequency_center,
        "tc": feat_settings.time_center,
        "g": feat_settings.gamma,
        "bin_per_octave": feat_settings.bins_per_octave
    }

    iters = enumerate(
        parallel_generator(
            _all_in_one_extract,
            data_pair,
            max_workers=num_threads,
            chunk_size=num_threads,
            label_extractor=label_extractor,
            t_unit=feat_settings.hop_size,
            **feat_extract_params
        )
    )
    for idx, ((feature, label), audio_idx) in iters:
        audio = data_pair[audio_idx][0]
        logger.info("Progress: %s/%s - %s", idx + 1, len(data_pair), audio)
        # print(f"Progress: {idx+1}/{len(data_pair)} - {audio}" + " "*6, end="\r")  # noqa: E226

        # Trim to the same length
        max_len = min(len(feature), len(label))
        feature = feature[:max_len]
        label = label[:max_len]

        basename = os.path.basename(audio)
        filename, _ = os.path.splitext(basename)
        out_hdf = jpath(out_path, filename + ".hdf")
        with h5py.File(out_hdf, "w") as out_f:
            out_f.create_dataset("feature", data=feature, compression="gzip", compression_opts=3)
            out_f.create_dataset("label", data=label, compression="gzip", compression_opts=3)


class VocalDatasetLoader(BaseDatasetLoader):
    """Dataset loader of 'vocal' module.

    Defines an additional parameter 'ctx_len' to detemine the context length
    of the input feature with repect to the current timestamp.
    """
    def __init__(self, ctx_len=9, feature_folder=None, feature_files=None, num_samples=100, slice_hop=1):
        super().__init__(
            feature_folder=feature_folder,
            feature_files=feature_files,
            num_samples=num_samples,
            slice_hop=slice_hop
        )
        self.ctx_len = ctx_len

    def _get_feature(self, hdf_name, slice_start):
        feat = self.hdf_refs[hdf_name]["feature"]

        pad_left = 0
        if slice_start - self.ctx_len < 0:
            pad_left = self.ctx_len - slice_start

        pad_right = 0
        if slice_start + self.ctx_len + 1 > len(feat):
            pad_right = slice_start + self.ctx_len + 1 - len(feat)

        start = max(slice_start - self.ctx_len, 0)
        end = min(slice_start + self.ctx_len + 1, len(feat))
        feat = feat[start:end]
        if (pad_left > 0) or (pad_right > 0):
            feat = np.pad(feat, ((pad_left, pad_right), (0, 0), (0, 0)))

        return feat  # Time x Freq x 9


if __name__ == "__main__":
    # pylint: disable=W0621,C0103
    app = VocalTranscription()
    tonas_train = "/data/TONAS/train_feature"
    mir1k_train = "/data/MIR-1K/train_feature"
    settings = VocalSettings()
    settings.training.steps = 1500
    settings.training.val_steps = 100
    settings.training.epoch = 20
    settings.model.shake_drop = False
    # app.train(feature_folder=tonas_train, semi_feature_folder=mir1k_train, model_name="test", vocal_settings=settings)
    audio = "/data/omnizart/checkpoints/ytd_audio_00105_TRFSJUR12903CB23E7.mp3.wav"
    audio = "/data/omnizart/checkpoints/Ava.wav"
    pred, feat = app.transcribe(audio, model_path="/data/omnizart/omnizart/checkpoints/vocal/vocal_test")
