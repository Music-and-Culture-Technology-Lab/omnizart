"""Application class of music.

Inludes core functions and interfaces for transcribing the audio, train
a model, generate feature of datasets, and evaluate on models.

See Also
--------
omnizart.base.BaseTranscription: The base class of all transcription/application classes.
"""

# pylint: disable=C0103,W0621,E0611
import os
from os.path import join as jpath
from datetime import datetime

import h5py
import tensorflow as tf

from omnizart.feature.wrapper_func import extract_cfp_feature
from omnizart.models.u_net import MultiHeadAttention, semantic_segmentation, semantic_segmentation_attn
from omnizart.music.inference import multi_inst_note_inference
from omnizart.music.prediction import predict
from omnizart.music.dataset import get_dataset
from omnizart.music.labels import (
    LabelType, MaestroLabelExtraction, MapsLabelExtraction, MusicNetLabelExtraction, PopLabelExtraction
)
from omnizart.music.losses import focal_loss, smooth_loss
from omnizart.base import BaseTranscription
from omnizart.utils import get_logger, dump_pickle, write_yaml, parallel_generator, ensure_path_exists
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
        feature = extract_cfp_feature(input_audio, harmonic=model_settings.feature.harmonic)

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

        if output is not None:
            save_to = output
            if os.path.isdir(save_to):
                save_to = jpath(save_to, os.path.basename(input_audio).replace(".wav", ".mid"))
            midi.write(save_to)
            logger.info("MIDI file has been written to %s", save_to)
        if os.environ.get("LOG_LEVEL", "") == "debug":
            dump_pickle({"pred": pred}, "./debug_pred.pickle")

        logger.info("Transcription finished")
        return midi

    def generate_feature(self, dataset_path, music_settings=None, num_threads=4):
        """Extract the feature of the whole dataset.

        To train the model, the first thing is to pre-process the data into feature
        representations. After downloading the dataset, use this function to generate
        the feature by giving the path to where the dataset stored, and the program
        will do all the rest of things.

        To specify the output path, modify the attribute
        ``music_settings.dataset.feature_save_path`` to the value you want.
        It will default to the folder under where the dataset stored, generating
        two folders: ``train_feature`` and ``test_feature``.

        Parameters
        ----------
        dataset_path: Path
            Path to the downloaded dataset.
        music_settings: MusicSettings
            The configuration instance that holds all relative settings for
            the life-cycle of building a model.
        num_threads:
            Number of threads for parallel extracting the features.

        See Also
        --------
        omnizart.constants.datasets:
            Supported dataset that can be applied and the split of training/testing pieces.
        """
        if music_settings is not None:
            assert isinstance(music_settings, MusicSettings)
            settings = music_settings
        else:
            settings = self.settings

        dataset_type = _resolve_dataset_type(dataset_path)
        if dataset_type is None:
            logger.warning(
                "The given path %s does not match any built-in processable dataset. Do nothing...",
                dataset_path
            )
            return
        logger.info("Inferred dataset type: %s", dataset_type)

        # Build instance mapping
        struct = {
            "maps": d_struct.MapsStructure(),
            "musicnet": d_struct.MusicNetStructure(),
            "maestro": d_struct.MaestroStructure(),
            "pop": d_struct.PopStructure()
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

        Train a new music model or continue to train on a pre-trained model.

        Parameters
        ----------
        feature_folder: Path
            Path to the generated feature.
        model_name: str
            The name of the trained model. If not given, will default to the
            current timestamp.
        input_model_path: Path
            Specify the path to the pre-trained model if you want to continue
            to fine-tune on the model.
        music_settings: MusicSettings
            The configuration instance that holds all relative settings for
            the life-cycle of building a model.
        """
        if music_settings is not None:
            assert isinstance(music_settings, MusicSettings)
            settings = music_settings
        else:
            settings = self.settings

        if input_model_path is not None:
            logger.info("Continue to train on model: %s", input_model_path)
            model, prev_set = self._load_model(input_model_path, custom_objects=self.custom_objects)
            settings.training.timesteps = prev_set.training.timesteps
            settings.training.label_type = prev_set.training.label_type
            settings.training.channels = prev_set.training.channels
            settings.model.save_path = prev_set.model.save_path

        logger.info("Using label type: %s", settings.training.label_type)
        l_type = LabelType(settings.training.label_type)

        logger.info("Constructing dataset instance")
        split = settings.training.steps / (settings.training.steps + settings.training.val_steps)
        train_feat_files, val_feat_files = get_train_val_feat_file_list(feature_folder, split=split)
        train_dataset = get_dataset(
            l_type.get_conversion_func(),
            feature_files=train_feat_files,
            batch_size=settings.training.batch_size,
            steps=settings.training.steps,
            timesteps=settings.training.timesteps,
            channels=[FEATURE_NAME_TO_NUMBER[ch_name] for ch_name in settings.training.channels],
            feature_num=settings.training.feature_num
        )
        val_dataset = get_dataset(
            l_type.get_conversion_func(),
            feature_files=val_feat_files,
            batch_size=settings.training.val_batch_size,
            steps=settings.training.val_steps,
            timesteps=settings.training.timesteps,
            channels=[FEATURE_NAME_TO_NUMBER[ch_name] for ch_name in settings.training.channels],
            feature_num=settings.training.feature_num
        )

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


def _resolve_dataset_type(dataset_path):
    low_path = os.path.basename(os.path.abspath(dataset_path)).lower()
    keywords = {"maps": "maps", "musicnet": "musicnet", "maestro": "maestro", "rhythm": "pop", "pop": "pop"}
    d_type = [val for key, val in keywords.items() if key in low_path]
    if len(d_type) == 0:
        return None

    assert len(set(d_type)) == 1
    return d_type[0]


def model_training_test():
    feature_folder = "/data/omnizart/tf_dataset_experiment/feature"

    settings = MusicSettings()
    settings.model.model_type = "aspp"
    settings.training.epoch = 3
    settings.training.steps = 15
    settings.training.val_steps = 15
    settings.training.batch_size = 8
    settings.training.val_batch_size = 8
    settings.training.timesteps = 128
    settings.training.label_type = "note-stream"
    settings.training.loss_function = "smooth"
    settings.training.early_stop = 1

    app = MusicTranscription()
    model_path, history = app.train(
        feature_folder, music_settings=settings, model_name="test2", input_model_path="checkpoints/music/music_test"
    )
    return model_path, history


if __name__ == "__main__":
    settings = MusicSettings()
    app = MusicTranscription()
    app.generate_feature("/media/data/maestro-v1.0.0")
