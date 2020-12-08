"""Application class of vocal-contour.

Inludes core functions and interfaces for frame-level vocal transcription: 
model training, feature pre-processing, and audio transcription.

See Also
--------
omnizart.base.BaseTranscription: The base class of all transcription/application classes.
"""

# pylint: disable=C0103,W0612,E0611,W0613
import os
from os.path import join as jpath
from datetime import datetime

import numpy as np
from scipy.io.wavfile import write as wavwrite
import h5py
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from mir_eval import sonify

from omnizart.base import BaseTranscription, BaseDatasetLoader
from omnizart.setting_loaders import VocalContourSettings
from omnizart.feature.wrapper_func import extract_cfp_feature
from omnizart.utils import get_logger, ensure_path_exists, parallel_generator
from omnizart.io import write_yaml
from omnizart.train import train_epochs, get_train_val_feat_file_list
from omnizart.callbacks import EarlyStopping, ModelCheckpoint
from omnizart.vocal_contour.inference import inference
from omnizart.models.utils import get_contour, padding
from omnizart.constants.datasets import MIR1KStructure
from omnizart.models.u_net import semantic_segmentation
from omnizart.music.losses import focal_loss


logger = get_logger("Vocal Contour")


class VocalContourTranscription(BaseTranscription):
    """Application class for vocal-contour transcription."""
    def __init__(self, conf_path=None):
        super().__init__(VocalContourSettings, conf_path=conf_path)

    def transcribe(self, input_audio, model_path=None, output="./"):
        """Transcribe frame-level fundamental frequency of vocal from the given audio.

        Parameters
        ----------
        input_audio: Path
            Path to the wav audio file.
        model_path: Path
            Path to the trained model or the transcription mode. If given a path, should be
            the folder that contains `arch.yaml`, `weights.h5`, and `configuration.yaml`.
        output: Path (optional)
            Path for writing out the extracted vocal f0. Default to current path.

        Returns
        -------
        f0: txt
            The transcribed f0 of the vocal contour in Hz.

        See Also
        --------
        omnizart.cli.vocal_contour.transcribe: The coressponding command line entry.
        """
        if not os.path.isfile(input_audio):
            raise FileNotFoundError(f"The given audio path does not exist. Path: {input_audio}")

        logger.info("Loading model...")
        model, model_settings = self._load_model(model_path)

        logger.info("Extracting feature...")
        feature = extract_cfp_feature(
            input_audio,
            hop=model_settings.feature.hop_size,
            win_size=model_settings.feature.window_size,
            down_fs=model_settings.feature.sampling_rate
        )

        logger.info("Predicting...")
        pred = inference(
            feature[:, :, 0],
            model,
            timestep=model_settings.training.timesteps
        )

        mat_contour = get_contour(pred)
        f0 = mat_contour[:, 1].astype(float)
        timestamp = np.arange(len(f0)) * model_settings.feature.hop_size
        wav = sonify.pitch_contour(timestamp, f0, model_settings.feature.sampling_rate)

        if output is not None:
            base = os.path.basename(input_audio)
            filename, ext = os.path.splitext(base)
            f0_out = f'{filename}_f0.txt'
            wav_trans = f'{filename}_trans.wav'
            save_to = output
            if os.path.isdir(save_to):
                f0_save_to = jpath(save_to, f0_out)
                wav_save_to = jpath(save_to, wav_trans)
            else:
                f0_save_to = f0_out
                wav_save_to = wav_trans
            np.savetxt(f0_save_to, f0)
            wavwrite(wav_save_to, model_settings.feature.sampling_rate, wav)
            logger.info("Text and Wav files have been written to %s", save_to)

        logger.info("Transcription finished")
        return f0

    def generate_feature(self, dataset_path, vocalcontour_settings=None, num_threads=4):
        """Extract the feature from the given dataset.

        To train the model, the first step is to pre-process the data into feature
        representations. After downloading the dataset, use this function to generate
        the feature by giving the path of the stored dataset.

        To specify the output path, modify the attribute
        ``vocalcontour_settings.dataset.feature_save_path`` (TODO: to confirm).
        It defaults to the folder of the stored dataset, and creates
        two folders: ``train_feature`` and ``test_feature``.

        Parameters
        ----------
        dataset_path: Path
            Path to the downloaded dataset.
        vocalcontour_settings: VocalContourSettings
            The configuration instance that holds all relative settings for
            the life-cycle of building a model.
        num_threads:
            Number of threads for parallel extraction of the feature.

        See Also
        --------
        omnizart.constants.datasets:
            The supported datasets and the corresponding training/testing splits.
        """
        settings = self._validate_and_get_settings(vocalcontour_settings)

        # Resolve feature output path
        train_feat_out_path, test_feat_out_path = self._resolve_feature_output_path(dataset_path, settings)
        logger.info("Output training feature to %s", train_feat_out_path)
        logger.info("Output testing feature to %s", test_feat_out_path)

        # Resolve dataset type (TODO: Implement MedleyDB)
        dataset_name, struct = _resolve_dataset_type(dataset_path)

        logger.info("Inferred dataset name: %s", dataset_name)
        label_paths = struct.get_train_labels(dataset_path)

        train_wavs = struct.get_train_wavs(dataset_path=dataset_path)
        logger.info(
            "Start extract training feature of the dataset. "
            "This may take time to finish and affect the computer's performance"
        )
        _parallel_feature_extraction(
            train_wavs, label_paths, train_feat_out_path, dataset_name, settings.feature, num_threads=num_threads
        )

        test_wavs = struct.get_test_wavs(dataset_path=dataset_path)
        logger.info(
            "Start extract testing feature of the dataset. "
            "This may take time to finish and affect the computer's performance"
        )
        _parallel_feature_extraction(
            test_wavs, label_paths, test_feat_out_path, dataset_name, settings.feature, num_threads=num_threads
        )

        # Writing out the settings
        write_yaml(settings.to_json(), jpath(train_feat_out_path, ".success.yaml"))
        write_yaml(settings.to_json(), jpath(test_feat_out_path, ".success.yaml"))
        logger.info("All done")

    def train(self, feature_folder, model_name=None, input_model_path=None, vocalcontour_settings=None):
        """Model training.

        Train the model from scratch or continue training given a model checkpoint.

        Parameters
        ----------
        feature_folder: Path
            Path to the generated feature.
        model_name: str
            The name of the trained model. If not given, will default to the
            current timestamp.
        input_model_path: Path
            Specify the path to the model checkpoint in order to fine-tune
            the model.
        vocalcontour_settings: VocalContourSettings
            The configuration that holds all relative settings for
            the life-cycle of model building.
        """
        settings = self._validate_and_get_settings(vocalcontour_settings)

        if input_model_path is not None:
            logger.info("Continue to train one model: %s", input_model_path)
            model, prev_set = self._load_model(input_model_path)
            settings.training.timesteps = prev_set.training.timesteps
            settings.model.save_path = prev_set.model.save_path

        logger.info("Constructing dataset instance")
        split = settings.training.steps / (settings.training.steps + settings.training.val_steps)
        train_feat_files, val_feat_files = get_train_val_feat_file_list(feature_folder, split=split)

        output_types = (tf.float32, tf.float32)
        train_dataset = VocalContourDatasetLoader(
            feature_files=train_feat_files,
            num_samples=settings.training.batch_size * settings.training.steps,
            timesteps=settings.training.timesteps
        ).get_dataset(settings.training.batch_size, output_types=output_types)

        val_dataset = VocalContourDatasetLoader(
            feature_files=val_feat_files,
            num_samples=settings.training.val_batch_size * settings.training.val_steps,
            timesteps=settings.training.timesteps
        ).get_dataset(settings.training.val_batch_size, output_types=output_types)

        if input_model_path is None:
            logger.info("Constructing new model")
            # NOTE: The default value of dropout rate for ConvBlock is different
            # in VocalSeg which is 0.2.
            model = semantic_segmentation(
                multi_grid_layer_n=1, feature_num=384, ch_num=1, timesteps=settings.training.timesteps
            )
        model.compile(optimizer="adam", loss=focal_loss, metrics=['accuracy'])

        logger.info("Resolving model output path")
        if model_name is None:
            model_name = str(datetime.now()).replace(" ", "_")
        if not model_name.startswith(settings.model.save_prefix):
            model_name = settings.model.save_prefix + "_" + model_name

        model_save_path = jpath(settings.model.save_path, model_name)
        ensure_path_exists(model_save_path)
        write_yaml(settings.to_json(), jpath(model_save_path, "configurations.yaml"))
        write_yaml(model.to_yaml(), jpath(model_save_path, "arch.yaml"), dump=False)
        logger.info("Model output to: %s", model_save_path)

        logger.info("Constructing callbacks")
        callbacks = [
            EarlyStopping(patience=settings.training.early_stop),
            ModelCheckpoint(model_save_path, save_weights_only=True)
        ]
        logger.info("Callback list: %s", callbacks)

        logger.info("Start training")
        history = train_epochs(
            model,
            train_dataset,
            validate_dataset=val_dataset,
            epochs=settings.training.epoch,
            steps=settings.training.steps,
            val_steps=settings.training.val_steps,
            callbacks=callbacks
        )

        return model_save_path, history


def _resolve_dataset_type(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"The given dataset directory does not exist. Path: {dataset_path}")

    low_path = os.path.basename(os.path.abspath(dataset_path)).lower()

    msg = "vocal_contour only accepts MIR-1K and MedleyDB as the training data."
    assert low_path in ['mir-1k', 'medleydb'], msg
    if low_path == 'medleydb':
        raise NotImplementedError("Using MedleyDB as the training data is no implemented yet.")

    target_dataset = MIR1KStructure

    return low_path, target_dataset


def _parallel_feature_extraction(wav_paths, label_paths, out_path, target_dataset, feat_settings, num_threads=4):
    assert target_dataset in ['medleydb', 'mir-1k'], \
        f"`target_dataset` should be either 'medleydb' or 'mir-1k'; got {target_dataset}"
    if target_dataset == "medleydb":
        raise NotImplementedError

    feat_extract_params = {
        "hop": feat_settings.hop_size,
        "down_fs": feat_settings.sampling_rate,
        "win_size": feat_settings.window_size
    }

    iters = enumerate(
        parallel_generator(
            extract_cfp_feature,
            wav_paths,
            max_workers=num_threads,
            use_thread=True,
            chunk_size=num_threads,
            **feat_extract_params
        )
    )

    fname_to_label_path = get_fn_to_path(label_paths)

    for idx, (feature, audio_idx) in iters:
        audio = wav_paths[audio_idx]

        print(f"Progress: {idx+1}/{len(wav_paths)} - {audio}" + " "*6, end="\r")  # noqa: E226

        # --------------------------------------------------------------------------
        # TODO: Perhaps imitate drum.app that puts `label_list` and `label_parser()`
        # in `parallel_generator()` to keep consistency
        audio_filename = get_filename(audio)
        label = fname_to_label_path[audio_filename]
        label_filename = get_filename(label)
        msg = f"{audio_idx}-th entry of the given `wav_paths` and `label_list` leads to\
            mismatched audio and label filenames: {audio_filename} and {label_filename}, respectively."
        assert audio_filename == label_filename, msg
        y = label_parser(label, target_dataset, vocal_track_list=None)
        filename = audio_filename
        # --------------------------------------------------------------------------

        out_hdf = jpath(out_path, filename + ".hdf")

        saved = False
        retry_times = 5
        for retry in range(retry_times):
            if saved:
                break
            try:
                with h5py.File(out_hdf, "w") as out_f:
                    out_f.create_dataset("feature", data=feature)
                    out_f.create_dataset("label", data=y)
                    saved = True
            except OSError as exp:
                logger.warning("OSError occurred, retrying %d times. Reason: %s", retry + 1, str(exp))
        if not saved:
            logger.error("H5py failed to save the feature file after %d retries.", retry_times)
            raise OSError
    print("")


# Brought from the original repo
# https://github.com/s603122001/Vocal-Melody-Extraction/blob/master/project/dataset_manage.py#L58
def label_parser(label, target_data, vocal_track_list=None):
    # parser label for mir1k
    if target_data == 'mir-1k':
        score = np.loadtxt(label)
        score_mat = np.zeros((len(score), 352))
        for i in range(score_mat.shape[0]):
            n = score[i]
            if n != 0:
                score_mat[i, int(np.round((score[i] - 21) * 4))] = 1
    # parser label for medleydb
    else:
        raise NotImplementedError

    return score_mat


class VocalContourDatasetLoader(BaseDatasetLoader):
    """Data loader for training the mdoel of ``vocal-contour``.

    Load feature and label for training. 

    Parameters
    ----------
    feature_folder: Path
        Path to the extracted feature files in `*.hdf`.
    feature_files: list[Path]
        List of path to the feature files in`*.hdf`.
    num_samples: int
        Total number of samples to yield.
    timesteps: int
        Time length of the feature.
    channels: list[int]
        Channels to be used for training. Allowed values are [1, 2, 3].
    feature_num: int
        Target size of feature dimension.
        Zero padding is done to resolve mismatched input and target size.

    Yields
    ------
    feature:
        Input features for model training.
    label:
        Coressponding labels.
    """
    def __init__(
        self,
        feature_folder=None,
        feature_files=None,
        num_samples=100,
        timesteps=128,
        channels=0,
        feature_num=384
    ):
        super().__init__(
            feature_folder=feature_folder, feature_files=feature_files, num_samples=num_samples, slice_hop=timesteps
        )

        self.feature_folder = feature_folder
        self.feature_files = feature_files
        self.num_samples = num_samples
        self.timesteps = timesteps
        self.channels = channels
        self.feature_num = feature_num

        self.hdf_refs = {}
        for hdf in self.hdf_files:
            ref = h5py.File(hdf, "r")
            self.hdf_refs[hdf] = ref

    def _get_feature(self, hdf_name, slice_start):
        feat = self.hdf_refs[hdf_name]["feature"]
        feat = feat[:, :, self.channels]
        feat = padding(feat, self.feature_num, self.slice_hop)
        feat = feat[slice_start:slice_start + self.slice_hop]
        return feat.reshape(self.timesteps, self.feature_num, 1)

    def _get_label(self, hdf_name, slice_start):
        label = self.hdf_refs[hdf_name]["label"]
        label = padding(label, self.feature_num, self.slice_hop)
        label = label[slice_start:slice_start + self.slice_hop]
        return to_categorical(label, num_classes=2)

    def _pre_yield(self, feature, label):
        feat_len = len(feature)
        label_len = len(label)

        if (feat_len == self.timesteps) and (label_len == self.timesteps):
            # All normal
            return feature, label

        # The length of feature and label are inconsistent. Trim to the same size as the shorter one.
        if feat_len > label_len:
            feature = feature[:label_len]
            feat_len = len(feature)
        else:
            label = label[:feat_len]
            label_len = len(label)

        return feature, label


def get_filename(filepath):
    basename = os.path.basename(filepath)
    filename, _ = os.path.splitext(basename)
    return filename


def get_fn_to_path(list_paths):
    fname_to_path = {}
    for path in list_paths:
        fname = get_filename(path)
        fname_to_path[fname] = path
    return fname_to_path
