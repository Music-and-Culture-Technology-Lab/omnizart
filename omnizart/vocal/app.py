import os
import glob
import shutil
import subprocess
from os.path import join as jpath
from collections import OrderedDict
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf
from spleeter.separator import Separator
from spleeter.utils.logging import logger as sp_logger

from omnizart.io import load_audio, write_yaml
from omnizart.utils import (
    get_logger, resolve_dataset_type, parallel_generator, ensure_path_exists, LazyLoader, get_filename
)
from omnizart.constants import datasets as d_struct
from omnizart.base import BaseTranscription, BaseDatasetLoader
from omnizart.feature.cfp import extract_vocal_cfp, _extract_vocal_cfp
from omnizart.setting_loaders import VocalSettings
from omnizart.vocal import labels as lextor
from omnizart.vocal.prediction import predict
from omnizart.vocal.inference import infer_interval, infer_midi
from omnizart.train import get_train_val_feat_file_list
from omnizart.models.pyramid_net import PyramidNet


logger = get_logger("Vocal Transcription")
vcapp = LazyLoader("vcapp", globals(), "omnizart.vocal_contour")


class SpleeterError(Exception):
    """Wrapper exception class around Spleeter errors"""
    pass


class VocalTranscription(BaseTranscription):
    """Application class for vocal note transcription.

    This application implements the training procedure in a semi-supervised way.
    """
    def __init__(self, conf_path=None):
        super().__init__(VocalSettings, conf_path=conf_path)

        # Disable logging information of Spleeter
        sp_logger.setLevel(40)  # logging.ERROR

    def transcribe(self, input_audio, model_path=None, output="./"):
        """Transcribe vocal notes in the audio.

        This function transcribes onset, offset, and pitch of the vocal in the audio.
        This module is reponsible for predicting onset and offset time of each note,
        and pitches are estimated by the `vocal-contour` submodule.

        Parameters
        ----------
        input_audio: Path
            Path to the raw audio file (.wav).
        model_path: Path
            Path to the trained model or the supported transcription mode.
        output: Path (optional)
            Path for writing out the transcribed MIDI file. Default to the current path.

        Returns
        -------
        midi: pretty_midi.PrettyMIDI
            The transcribed vocal notes.

        Outputs
        -------
        This function will outputs three files as listed below:

        - <song>.mid: the MIDI file with complete transcription results in piano sondfount.
        - <song>_f0.csv: pitch contour information of the vocal.
        - <song>_trans.wav: the rendered pitch contour audio.

        See Also
        --------
        omnizart.cli.vocal.transcribe: CLI entry point of this function.
        omnizart.vocal_contour.transcribe: Pitch estimation function.
        """
        logger.info("Separating vocal track from the audio...")
        command = ["spleeter", "separate", input_audio, "-o", "./"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, error = process.communicate()
        if process.returncode != 0:
            raise SpleeterError(error.decode("utf-8"))

        # Resolve the path of separated output files
        folder_path = jpath("./", get_filename(input_audio))
        vocal_wav_path = jpath(folder_path, "vocals.wav")
        wav, fs = load_audio(vocal_wav_path)

        # Clean out the output files
        shutil.rmtree(folder_path)

        logger.info("Loading model...")
        model, model_settings = self._load_model(model_path)

        logger.info("Extracting feature...")
        feature = _extract_vocal_cfp(
            wav,
            fs,
            down_fs=model_settings.feature.sampling_rate,
            hop=model_settings.feature.hop_size,
            fr=model_settings.feature.frequency_resolution,
            fc=model_settings.feature.frequency_center,
            tc=model_settings.feature.time_center,
            g=model_settings.feature.gamma,
            bin_per_octave=model_settings.feature.bins_per_octave
        )

        logger.info("Predicting...")
        pred = predict(feature, model)

        logger.info("Infering notes...")
        interval = infer_interval(
            pred,
            ctx_len=model_settings.inference.context_length,
            threshold=model_settings.inference.threshold,
            min_dura=model_settings.inference.min_duration,
            t_unit=model_settings.feature.hop_size
        )

        logger.info("Extracting pitch contour")
        agg_f0 = vcapp.app.transcribe(input_audio, model_path=model_settings.inference.pitch_model, output=output)

        logger.info("Inferencing MIDI...")
        midi = infer_midi(interval, agg_f0, t_unit=model_settings.feature.hop_size)

        self._output_midi(output=output, input_audio=input_audio, midi=midi)
        logger.info("Transcription finished")
        return midi

    def generate_feature(self, dataset_path, vocal_settings=None, num_threads=4):
        """Extract the feature of the whole dataset.

        Currently supports MIR-1K and TONAS datasets. To train the model, you have to prepare the training
        data first, then process it into feature representations. After downloading the dataset,
        use this function to do the pre-processing and transform the raw data into features.

        To specify the output path, modify the attribute
        ``vocal_settings.dataset.feature_save_path`` to the value you want.
        It will default to the folder under where the dataset stored, generating
        two folders: ``train_feature`` and ``test_feature``.

        Parameters
        ----------
        dataset_path: Path
            Path to the downloaded dataset.
        vocal_settings: VocalSettings
            The configuration instance that holds all relative settings for
            the life-cycle of building a model.
        num_threads:
            Number of threads for parallel extracting the features.
        """
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
        """Model training.

        Train a new model or continue to train on a previously trained model.

        Parameters
        ----------
        feature_folder: Path
            Path to the folder containing generated feature.
        semi_feature_folder: Path
            If specified, semi-supervise learning will be leveraged, and the feature
            files contained in this folder will be used as unsupervised data.
        model_name: str
            The name for storing the trained model. If not given, will default to the
            current timesamp.
        input_model_path: Path
            Continue to train on the pre-trained model by specifying the path.
        vocal_settings: VocalSettings
            The configuration instance that holds all relative settings for
            the life-cycle of building a model.
        """
        settings = self._validate_and_get_settings(vocal_settings)

        if input_model_path is not None:
            logger.info("Continue to train on model: %s", input_model_path)
            model, prev_set = self._load_model(input_model_path)
            settings.model.save_path = prev_set.model.save_path

        logger.info("Constructing dataset instance")
        split = settings.training.steps / (settings.training.steps + settings.training.val_steps)
        train_feat_files, val_feat_files = get_train_val_feat_file_list(feature_folder, split=split)

        output_types = (tf.float32, tf.float32)
        output_shapes = ((settings.training.context_length*2 + 1, 174, 9), (19, 6))  # noqa: E226
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

        # Notice: the original implementation uses AdamW as the optimizer, which is also viable through
        # tensorflow_addons.optimizers.AdamW. However we found that by using AdamW, the model would fail
        # to converge, and instead the training loss will get higher and higher.
        optimizer = tf.keras.optimizers.Adam(learning_rate=settings.training.init_learning_rate)
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
        separator._params["stft_backend"] = "librosa"  # pylint: disable=protected-access
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

    Defines an additional parameter 'ctx_len' to determine the context length
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

    def _get_label(self, hdf_name, slice_start):
        label = self.hdf_refs[hdf_name]["label"]

        pad_left = 0
        if slice_start - self.ctx_len < 0:
            pad_left = self.ctx_len - slice_start

        pad_right = 0
        if slice_start + self.ctx_len + 1 > len(label):
            pad_right = slice_start + self.ctx_len + 1 - len(label)

        start = max(slice_start - self.ctx_len, 0)
        end = min(slice_start + self.ctx_len + 1, len(label))
        label = label[start:end]
        if (pad_left > 0) or (pad_right > 0):
            label = np.pad(label, ((pad_left, pad_right), (0, 0)))

        return label  # Time x 6
