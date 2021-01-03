"""Application class of music.

Inludes core functions and interfaces for music transcription:
model training, feature pre-processing, and audio transcription.

See Also
--------
omnizart.base.BaseTranscription: The base class of all transcription/application classes.
"""

# pylint: disable=C0103,W0621,E0611
import os
import pickle
from os.path import join as jpath
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf

from omnizart.feature.wrapper_func import extract_cfp_feature
from omnizart.models.u_net import MultiHeadAttention, semantic_segmentation, semantic_segmentation_attn
from omnizart.music.inference import multi_inst_note_inference
from omnizart.music.prediction import predict
from omnizart.music.labels import (
    LabelType, MaestroLabelExtraction, MapsLabelExtraction, MusicNetLabelExtraction, PopLabelExtraction
)
from omnizart.music.losses import focal_loss, smooth_loss
from omnizart.base import BaseTranscription, BaseDatasetLoader
from omnizart.utils import get_logger, parallel_generator, ensure_path_exists, resolve_dataset_type
from omnizart.io import dump_pickle, write_yaml
from omnizart.train import train_epochs, get_train_val_feat_file_list
from omnizart.callbacks import EarlyStopping, ModelCheckpoint
from omnizart.setting_loaders import MusicSettings
from omnizart.constants.midi import MUSICNET_INSTRUMENT_PROGRAMS, POP_INSTRUMENT_PROGRAMES
from omnizart.constants.feature import FEATURE_NAME_TO_NUMBER
import omnizart.constants.datasets as d_struct

logger = get_logger("Music Transcription")


class MusicTranscription(BaseTranscription):
    """Application class for music transcription.

    Inherited from the BaseTranscription class to make sure everything
    needed got override.
    """
    def __init__(self, conf_path=None):
        super().__init__(MusicSettings, conf_path=conf_path)
        self.mode_inst_mapping = {
            "Piano": MUSICNET_INSTRUMENT_PROGRAMS,
            "Stream": MUSICNET_INSTRUMENT_PROGRAMS,
            "Pop": POP_INSTRUMENT_PROGRAMES
        }
        self.label_trans_mode_mapping = {
            "note": "Piano",
            "frame": "Piano",
            "true-frame": "Piano",
            "note-stream": "Stream",
            "frame-stream": "Stream",
            "true-frame-stream": "Stream",
            "pop-note-stream": "Pop"
        }
        self.custom_objects = {"MultiHeadAttention": MultiHeadAttention}

    def transcribe(self, input_audio, model_path=None, output="./"):
        """Transcribe notes and instruments of the given audio.

        This function transcribes notes (onset, duration) of each instruments in the audio.
        The results will be written out as a MIDI file.

        Parameters
        ----------
        input_audio: Path
            Path to the wav audio file.
        model_path: Path
            Path to the trained model or the transcription mode. If given a path, should be
            the folder that contains `arch.yaml`, `weights.h5`, and `configuration.yaml`.
        output: Path (optional)
            Path for writing out the transcribed MIDI file. Default to current path.

        Returns
        -------
        midi: pretty_midi.PrettyMIDI
            The transcribed notes of different instruments.

        See Also
        --------
        omnizart.cli.music.transcribe: The coressponding command line entry.
        """
        if not os.path.isfile(input_audio):
            raise FileNotFoundError(f"The given audio path does not exist. Path: {input_audio}")

        logger.info("Loading model...")
        model, model_settings = self._load_model(model_path, custom_objects=self.custom_objects)

        logger.info("Extracting feature...")
        feature = extract_cfp_feature(
            input_audio,
            down_fs=model_settings.feature.sampling_rate,
            hop=model_settings.feature.hop_size,
            win_size=model_settings.feature.window_size,
            fr=model_settings.feature.frequency_resolution,
            fc=model_settings.feature.frequency_center,
            tc=model_settings.feature.time_center,
            g=model_settings.feature.gamma,
            bin_per_octave=model_settings.feature.bins_per_octave,
            harmonic_num=model_settings.feature.harmonic_number,
            harmonic=model_settings.feature.harmonic
        )

        logger.info("Predicting...")
        channels = [FEATURE_NAME_TO_NUMBER[ch_name] for ch_name in model_settings.training.channels]
        pred = predict(
            feature[:, :, channels],
            model,
            timesteps=model_settings.training.timesteps,
            feature_num=model_settings.training.feature_num
        )

        logger.info("Infering notes....")
        midi = multi_inst_note_inference(
            pred,
            mode=model_settings.training.label_type,
            onset_th=model_settings.inference.onset_th,
            dura_th=model_settings.inference.dura_th,
            frm_th=model_settings.inference.frame_th,
            inst_th=model_settings.inference.inst_th,
            t_unit=model_settings.feature.hop_size,
            channel_program_mapping=self.mode_inst_mapping[model_settings.transcription_mode],
        )

        self._output_midi(output=output, input_audio=input_audio, midi=midi)
        if os.environ.get("LOG_LEVEL", "") == "debug":
            dump_pickle({"pred": pred}, "./debug_pred.pickle")

        logger.info("Transcription finished")
        return midi

    def generate_feature(self, dataset_path, music_settings=None, num_threads=4):
        """Extract the feature from the given dataset.

        To train the model, the first step is to pre-process the data into feature
        representations. After downloading the dataset, use this function to generate
        the feature by giving the path of the stored dataset.

        To specify the output path, modify the attribute
        ``music_settings.dataset.feature_save_path``.
        It defaults to the folder under where the dataset stored, generating
        two folders: ``train_feature`` and ``test_feature``.

        Parameters
        ----------
        dataset_path: Path
            Path to the downloaded dataset.
        music_settings: MusicSettings
            The configuration instance that holds all relative settings for
            the life-cycle of building a model.
        num_threads:
            Number of threads for parallel extraction the feature.

        See Also
        --------
        omnizart.constants.datasets:
            The supported datasets and the corresponding training/testing splits.
        """
        settings = self._validate_and_get_settings(music_settings)

        dataset_type = resolve_dataset_type(
            dataset_path,
            keywords={"maps": "maps", "musicnet": "musicnet", "maestro": "maestro", "rhythm": "pop", "pop": "pop"}
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
            "maps": d_struct.MapsStructure,
            "musicnet": d_struct.MusicNetStructure,
            "maestro": d_struct.MaestroStructure,
            "pop": d_struct.PopStructure
        }[dataset_type]
        label_extractor = {
            "maps": MapsLabelExtraction,
            "musicnet": MusicNetLabelExtraction,
            "maestro": MaestroLabelExtraction,
            "pop": PopLabelExtraction
        }[dataset_type]

        # Fetching wav files
        train_wav_files = struct.get_train_wavs(dataset_path=dataset_path)
        test_wav_files = struct.get_test_wavs(dataset_path=dataset_path)
        logger.info("Number of total training wavs: %d", len(train_wav_files))
        logger.info("Number of total testing wavs: %d", len(test_wav_files))

        # Resolve feature output path
        train_feat_out_path, test_feat_out_path = self._resolve_feature_output_path(dataset_path, settings)
        logger.info("Output training feature to %s", train_feat_out_path)
        logger.info("Output testing feature to %s", test_feat_out_path)

        # Feature extraction
        logger.info(
            "Start extract the feature of the dataset %s. "
            "This may take time to finish and affect the computer's performance.",
            dataset_type.title()
        )
        logger.info("Extracting training feature")
        _parallel_feature_extraction(train_wav_files, train_feat_out_path, settings.feature, num_threads=num_threads)
        logger.info("Extracting testing feature")
        _parallel_feature_extraction(test_wav_files, test_feat_out_path, settings.feature, num_threads=num_threads)
        logger.info("Extraction finished")

        # Fetching label files
        train_label_files = struct.get_train_labels(dataset_path=dataset_path)
        test_label_files = struct.get_test_labels(dataset_path=dataset_path)
        logger.info("Number of total training labels: %d", len(train_label_files))
        logger.info("Number of total testing labels: %d", len(test_label_files))
        assert len(train_label_files) == len(train_wav_files)
        assert len(test_label_files) == len(test_wav_files)

        # Extract labels
        logger.info("Start extracting the label of the dataset %s", dataset_type.title())
        label_extractor.process(train_label_files, out_path=train_feat_out_path, t_unit=settings.feature.hop_size)
        label_extractor.process(test_label_files, out_path=test_feat_out_path, t_unit=settings.feature.hop_size)

        # Writing out the settings
        write_yaml(settings.to_json(), jpath(train_feat_out_path, ".success.yaml"))
        write_yaml(settings.to_json(), jpath(test_feat_out_path, ".success.yaml"))
        logger.info("All done")

    def train(self, feature_folder, model_name=None, input_model_path=None, music_settings=None):
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
        music_settings: MusicSettings
            The configuration that holds all relative settings for
            the life-cycle of model building.
        """
        settings = self._validate_and_get_settings(music_settings)

        if input_model_path is not None:
            logger.info("Continue to train on model: %s", input_model_path)
            model, prev_set = self._load_model(input_model_path, custom_objects=self.custom_objects)
            settings.training.timesteps = prev_set.training.timesteps
            settings.training.label_type = prev_set.training.label_type
            settings.training.channels = prev_set.training.channels
            settings.model.save_path = prev_set.model.save_path
            settings.transcription_mode = prev_set.transcription_mode

        logger.info("Using label type: %s", settings.training.label_type)
        l_type = LabelType(settings.training.label_type)
        settings.transcription_mode = self.label_trans_mode_mapping[settings.training.label_type]

        logger.info("Constructing dataset instance")
        split = settings.training.steps / (settings.training.steps + settings.training.val_steps)
        train_feat_files, val_feat_files = get_train_val_feat_file_list(feature_folder, split=split)

        output_types = (tf.float32, tf.float32)
        train_dataset = MusicDatasetLoader(
                l_type.get_conversion_func(),
                feature_files=train_feat_files,
                num_samples=settings.training.batch_size * settings.training.steps,
                timesteps=settings.training.timesteps,
                channels=[FEATURE_NAME_TO_NUMBER[ch_name] for ch_name in settings.training.channels],
                feature_num=settings.training.feature_num
            ) \
            .get_dataset(settings.training.batch_size, output_types=output_types)
        val_dataset = MusicDatasetLoader(
                l_type.get_conversion_func(),
                feature_files=val_feat_files,
                num_samples=settings.training.val_batch_size * settings.training.val_steps,
                timesteps=settings.training.timesteps,
                channels=[FEATURE_NAME_TO_NUMBER[ch_name] for ch_name in settings.training.channels],
                feature_num=settings.training.feature_num
            ) \
            .get_dataset(settings.training.val_batch_size, output_types=output_types)

        if input_model_path is None:
            logger.info("Creating new model with type: %s", settings.model.model_type)
            model_func = {
                "aspp": semantic_segmentation,
                "attn": semantic_segmentation_attn
            }[settings.model.model_type]
            model = model_func(
                timesteps=settings.training.timesteps,
                out_class=l_type.get_out_classes(),
                ch_num=len(settings.training.channels)
            )

        logger.info("Compiling model with loss function type: %s", settings.training.loss_function)
        loss_func = {
            "smooth": lambda y, x: smooth_loss(y, x, total_chs=l_type.get_out_classes()),
            "focal": focal_loss,
            "bce": tf.keras.losses.BinaryCrossentropy()
        }[settings.training.loss_function]
        model.compile(optimizer="adam", loss=loss_func, metrics=['accuracy'])

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


def _parallel_feature_extraction(audio_list, out_path, feat_settings, num_threads=4):
    feat_extract_params = {
        "hop": feat_settings.hop_size,
        "win_size": feat_settings.window_size,
        "fr": feat_settings.frequency_resolution,
        "fc": feat_settings.frequency_center,
        "tc": feat_settings.time_center,
        "g": feat_settings.gamma,
        "bin_per_octave": feat_settings.bins_per_octave,
        "harmonic_num": feat_settings.harmonic_number
    }

    iters = enumerate(
        parallel_generator(
            extract_cfp_feature,
            audio_list,
            max_workers=num_threads,
            use_thread=True,
            chunk_size=num_threads,
            harmonic=feat_settings.harmonic,
            **feat_extract_params
        )
    )
    for idx, (feature, audio_idx) in iters:
        audio = audio_list[audio_idx]
        # logger.info("Progress: %s/%s - %s", idx+1, len(audio_list), audio)
        print(f"Progress: {idx+1}/{len(audio_list)} - {audio}" + " "*6, end="\r")  # noqa: E226

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
                    out_f.create_dataset("feature", data=feature)
                    saved = True
            except OSError as exp:
                logger.warning("OSError occurred, retrying %d times. Reason: %s", retry + 1, str(exp))
        if not saved:
            logger.error("H5py failed to save the feature file after %d retries.", retry_times)
            raise OSError
    print("")


class MusicDatasetLoader(BaseDatasetLoader):
    """Data loader for training the mdoel of ``music``.

    Load feature and label for training. Also converts the custom format of
    label into piano roll representation.

    Parameters
    ----------
    label_conversion_func: callable
        The function that will be used for converting the customized label format
        into numpy array.
    feature_folder: Path
        Path to the extracted feature files, including `*.hdf` and `*.pickle` pairs,
        which refers to feature and label files, respectively.
    feature_files: list[Path]
        List of path of `*.hdf` feature files. Corresponding label files should also
        under the same folder.
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
        Corresponding labels.
    """
    def __init__(
        self,
        label_conversion_func,
        feature_folder=None,
        feature_files=None,
        num_samples=100,
        timesteps=128,
        channels=[1, 3],
        feature_num=352
    ):
        super().__init__(
            feature_folder=feature_folder, feature_files=feature_files, num_samples=num_samples, slice_hop=timesteps
        )

        self.conv_func = label_conversion_func
        self.feature_folder = feature_folder
        self.feature_files = feature_files
        self.num_samples = num_samples
        self.timesteps = timesteps
        self.channels = channels

        self.hdf_refs = {}
        self.pkls = {}
        for hdf in self.hdf_files:
            ref = h5py.File(hdf, "r")
            self.hdf_refs[hdf] = ref
            self.pkls[hdf] = pickle.load(open(hdf.replace(".hdf", ".pickle"), "rb"))

        ori_feature_num = list(self.hdf_refs.values())[0]["feature"]
        diff = feature_num - ori_feature_num.shape[1]
        pad_b = diff // 2
        pad_t = diff - pad_b
        self.pad = ((pad_b != 0) or (pad_t != 0))
        self.pad_shape = ((0, 0), (pad_b, pad_t), (0, 0))

    def _get_feature(self, hdf_name, slice_start):
        feat = self.hdf_refs[hdf_name]["feature"]
        feat = feat[slice_start:slice_start + self.slice_hop, :, self.channels].squeeze()
        if self.pad:
            feat = np.pad(feat, self.pad_shape, constant_values=0)
        return feat

    def _get_label(self, hdf_name, slice_start):
        ll = self.pkls[hdf_name][slice_start:slice_start + self.slice_hop]
        label = self.conv_func(ll)
        if self.pad:
            label = np.pad(label, self.pad_shape, constant_values=0)
        return label

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

        # Padding zeros to the end if the squence length is shorter than 'timesteps'.
        if feat_len != self.timesteps:
            assert feat_len < self.timesteps
            diff = self.timesteps - feat_len
            feature = np.pad(feature, ((0, diff), (0, 0), (0, 0)))
        if label_len != self.timesteps:
            assert label_len < self.timesteps
            diff = self.timesteps - label_len
            label = np.pad(label, ((0, diff), (0, 0), (0, 0)))

        return feature, label


if __name__ == "__main__":
    settings = MusicSettings()
    app = MusicTranscription()
    app.generate_feature("/media/data/maestro-v1.0.0")
