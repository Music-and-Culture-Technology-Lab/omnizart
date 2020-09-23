"""Application class of music.

Inludes core functions and interfaces for transcribing the audio, train
a model, generate feature of datasets, and evaluate on models.

See Also
--------
omnizart.base.BaseTranscription: The base class of all transcription/application classes.
"""

# pylint: disable=C0103,W0612,E0611
import os
import glob
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from scipy.special import expit

from omnizart.feature.cfp import extract_cfp
from omnizart.feature.hcfp import extract_hcfp
from omnizart.models.u_net import MultiHeadAttention, semantic_segmentation, semantic_segmentation_attn
from omnizart.music.inference import multi_inst_note_inference
from omnizart.music.utils import create_batches, cut_batch_pred, cut_frame
from omnizart.music.dataset import get_dataset
from omnizart.music.labels import LabelType
from omnizart.music.losses import focal_loss, smooth_loss
from omnizart.base import BaseTranscription
from omnizart.utils import get_logger, dump_pickle, load_yaml, write_yaml
from omnizart.train import train_epochs
from omnizart.callbacks import EarlyStopping, ModelCheckpoint
from omnizart.setting_loaders import MusicSettings
from omnizart.constants.midi import MUSICNET_INSTRUMENT_PROGRAMS, POP_INSTRUMENT_PROGRAMES
from omnizart.constants.feature import FEATURE_NAME_TO_NUMBER

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
        model, model_settings = self._load_model(model_path, custom_objects=self.custom_objects)

        logger.info("Extracting feature...")
        if model_settings.feature.harmonic:
            spec, gcos, ceps, cenf = extract_hcfp(input_audio)
            feature = np.dstack([spec, gcos, ceps])
        else:
            z, spec, gcos, ceps, cenf = extract_cfp(input_audio)
            feature = np.dstack([z.T, spec.T, gcos.T, ceps.T])

        logger.info("Predicting...")
        channels = [FEATURE_NAME_TO_NUMBER[ch_name] for ch_name in model_settings.training.channels]
        pred = self._predict(feature[:, :, channels], model, timesteps=model_settings.training.timesteps)

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
            save_to = os.path.join(output, os.path.basename(input_audio).replace(".wav", ".mid"))
            midi.write(save_to)
            logger.info("MIDI file has been written to %s", save_to)
        if os.environ["LOG_LEVEL"] == "debug":
            dump_pickle({"pred": pred}, os.path.join(save_to, "debug_pred.pickle"))
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

        # Initiate lamda function for latter processing of prediction
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

    def train(self, feature_folder, model_name=None, input_model_path=None, music_settings=None):
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
        train_feat_files, val_feat_files = self._get_train_val_feat_file_list(feature_folder, split=split)
        train_dataset = get_dataset(
            l_type.get_conversion_func(),
            feature_files=train_feat_files,
            batch_size=settings.training.batch_size,
            steps=settings.training.steps,
            timesteps=settings.training.timesteps,
            channels=[FEATURE_NAME_TO_NUMBER[ch_name] for ch_name in settings.training.channels]
        )
        val_dataset = get_dataset(
            l_type.get_conversion_func(),
            feature_files=val_feat_files,
            batch_size=settings.training.val_batch_size,
            steps=settings.training.val_steps,
            timesteps=settings.training.timesteps,
            channels=[FEATURE_NAME_TO_NUMBER[ch_name] for ch_name in settings.training.channels]
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
            model_save_path = os.path.join(settings.model.save_path, model_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        write_yaml(settings.to_json(), os.path.join(model_save_path, "configurations.yaml"))
        write_yaml(model.to_yaml(), os.path.join(model_save_path, "arch.yaml"), dump=False)

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

    def _get_train_val_feat_file_list(self, feature_folder, split=0.9):
        feat_files = glob.glob(f"{feature_folder}/*.hdf")
        sidx = round(len(feat_files)*split)
        random.shuffle(feat_files)
        train_files = feat_files[:sidx]
        val_files = feat_files[sidx:]
        return train_files, val_files


if __name__ == "__main__":
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

