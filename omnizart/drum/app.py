# pylint: disable=C0103,W0612,E0611
import os
from os.path import join as jpath
import time
from datetime import datetime
import asyncio

import h5py
import tensorflow as tf

from omnizart.feature.wrapper_func import extract_patch_cqt
from omnizart.drum.prediction import predict
from omnizart.drum.labels import extract_label_13_inst
from omnizart.drum.dataset import get_dataset
from omnizart.models.spectral_norm_net import drum_model, ConvSN2D
from omnizart.utils import get_logger, ensure_path_exists, parallel_generator, write_yaml
from omnizart.base import BaseTranscription
from omnizart.setting_loaders import DrumSettings
from omnizart.train import train_epochs, get_train_val_feat_file_list
from omnizart.callbacks import EarlyStopping, ModelCheckpoint
from omnizart.constants.datasets import PopStructure
from omnizart.constants.feature import NOTE_PRIORITY_ARRAY


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

    def generate_feature(self, dataset_path, drum_settings=None, num_threads=3):
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
        _parallel_feature_extraction_v2(
            train_wavs, train_labels, train_feat_out_path, settings.feature, num_threads=num_threads
        )

        test_wavs = struct.get_test_wavs(dataset_path=dataset_path)
        test_labels = struct.get_test_labels(dataset_path=dataset_path)
        logger.info(
            "Start extract testing feature of the dataset. "
            "This may take time to finish and affect the computer's performance"
        )
        assert len(test_wavs) == len(test_labels)
        _parallel_feature_extraction_v2(
            test_wavs, test_labels, test_feat_out_path, settings.feature, num_threads=num_threads
        )

        # Writing out the settings
        write_yaml(settings.to_json(), jpath(train_feat_out_path, ".success.yaml"))
        write_yaml(settings.to_json(), jpath(test_feat_out_path, ".success.yaml"))
        logger.info("All done")

    def train(self, feature_folder, model_name=None, input_model_path=None, drum_settings=None):
        if drum_settings is not None:
            assert isinstance(drum_settings, DrumSettings)
            settings = drum_settings
        else:
            settings = self.settings

        if input_model_path is not None:
            logger.info("Continue to train on model: %s", input_model_path)
            model, prev_set = self._load_model(input_model_path, custom_objects=self.custom_objects)
            settings.model.save_path = prev_set.model.save_path
            settings.training.init_learninig_rate = prev_set.training.init_learning_rate
            settings.training.res_block_num = prev_set.training.res_block_num

        logger.info("Constructing dataset instance")
        split = settings.training.steps / (settings.training.steps + settings.training.val_steps)
        train_feat_files, val_feat_files = get_train_val_feat_file_list(feature_folder, split=split)
        train_dataset = get_dataset(
            feature_files=train_feat_files,
            batch_size=settings.training.batch_size,
            steps=settings.training.steps
        )
        val_dataset = get_dataset(
            feature_files=val_feat_files,
            batch_size=settings.training.val_batch_size,
            steps=settings.training.val_steps
        )

        if input_model_path is None:
            logger.info("Constructing new model")
            model = drum_model(
                out_classes=13,
                mini_beat_per_seg=settings.feature.mini_beat_per_segment,
                res_block_num=settings.training.res_block_num
            )

        optimizer = tf.keras.optimizers.Adam(learning_rate=settings.training.init_learning_rate)
        model.compile(optimizer=optimizer, loss=_loss_func, metrics=["accuracy"])

        logger.info("Resolving model output path")
        if model_name is None:
            model_name = str(datetime.now()).replace(" ", "_")
        if not model_name.startswith(settings.model.save_prefix):
            model_name = settings.model.save_prefix + "_" + model_name
            model_save_path = jpath(settings.model.save_path, model_name)
        ensure_path_exists(model_save_path)
        write_yaml(settings.to_json(), jpath(model_save_path, "configurations.yaml"))
        write_yaml(model.to_yaml(), jpath(model_save_path, "arch.yaml"), dump=False)

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


def _parallel_feature_extraction(wav_paths, label_paths, out_path, feat_settings, num_threads=3):
    label_path_mapping = _gen_wav_label_path_mapping(label_paths)
    iters = enumerate(
        parallel_generator(
            _all_in_one_extract,
            wav_paths,
            max_workers=num_threads,
            use_thread=True,
            chunk_size=num_threads,
            label_path_mapping=label_path_mapping,
            feat_settings=feat_settings
        )
    )
    for idx, ((patch_cqt, m_beat_arr, label_128, label_13), audio_idx) in iters:
        audio = wav_paths[audio_idx]
        # print(f"Progress: {idx+1}/{len(wav_paths)} - {audio}" + " "*6, end="\r")  # noqa: E226
        logger.info("Progress: %d/%d - %s", idx+1, len(wav_paths), audio)  # noqa: E226

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


def _parallel_feature_extraction_v2(wav_paths, label_paths, out_path, feat_settings, num_threads=5):
    iter_num = len(wav_paths) / num_threads
    if int(iter_num) < iter_num:
        iter_num += 1
    iter_num = int(iter_num)

    label_path_mapping = _gen_wav_label_path_mapping(label_paths)
    for iter_idx in range(iter_num):
        loop = asyncio.get_event_loop()
        tasks = []
        for chunk in range(num_threads):
            wav_idx = num_threads*iter_idx + chunk  # noqa: E226
            if wav_idx >= len(wav_paths):
                break
            logger.info("%s/%s - %s", wav_idx+1, len(wav_paths), wav_paths[wav_idx])  # noqa: E226
            tasks.append(
                loop.create_task(_async_all_in_one_extract(
                    wav_paths[wav_idx], label_path_mapping, feat_settings
                ))
            )

        group = asyncio.gather(*tasks, return_exceptions=True)
        print("Waiting...")
        results = loop.run_until_complete(group)
        for result in results:
            patch_cqt, m_beat_arr, label_128, label_13, wav_path = result
            basename = os.path.basename(wav_path)
            filename, _ = os.path.splitext(basename)
            out_hdf = jpath(out_path, filename + ".hdf")
            with h5py.File(out_hdf, "w") as out_f:
                out_f.create_dataset("feature", data=patch_cqt, compression="gzip", compression_opts=3)
                out_f.create_dataset("label", data=label_13, compression="gzip", compression_opts=3)
                out_f.create_dataset("label_128", data=label_128, compression="gzip", compression_opts=3)
                out_f.create_dataset("mini_beat_arr", data=m_beat_arr, compression="gzip", compression_opts=3)


async def _async_all_in_one_extract(wav_path, label_path_mapping, feat_settings):
    loop = asyncio.get_event_loop()
    patch_cqt, m_beat_arr, label_128, label_13 = await loop.run_in_executor(
        None, _all_in_one_extract, wav_path, label_path_mapping, feat_settings
    )
    return patch_cqt, m_beat_arr, label_128, label_13, wav_path


def _all_in_one_extract(wav_path, label_path_mapping, feat_settings):
    patch_cqt, m_beat_arr = extract_patch_cqt(
        wav_path, sampling_rate=feat_settings.sampling_rate, hop_size=feat_settings.hop_size
    )

    label_path = label_path_mapping[os.path.basename(wav_path)]
    label_128, label_13 = extract_label_13_inst(label_path, m_beat_arr)
    return patch_cqt, m_beat_arr, label_128, label_13


def _gen_wav_label_path_mapping(label_paths):
    mapping = {}
    for label_path in label_paths:
        f_name = os.path.basename(label_path).replace(".mid", ".wav")
        wav_name = f_name.replace("align_mid", "ytd_audio")
        mapping[wav_name] = label_path
    return mapping


def _loss_func(target, pred, soft_loss_range=20):
    recon_error = tf.abs(target - pred)
    recon_error_soft = tf.compat.v1.where(
        recon_error <= soft_loss_range * tf.ones_like(recon_error),
        tf.zeros_like(recon_error),
        recon_error - soft_loss_range * tf.ones_like(recon_error)
    )

    # shape = shape_list(recon_error_soft[:, :, :, :])
    # note_priority_arr = tf.constant(NOTE_PRIORITY_ARRAY, dtype=recon_error.dtype)
    # note_priority_ary_in_expanded = note_priority_arr + tf.zeros(shape, dtype=note_priority_arr.dtype)
    # recon_error_soft_priority = tf.multiply(recon_error_soft, note_priority_ary_in_expanded)
    # recon_error_soft_flat = tf.reshape(
    #     recon_error_soft_priority,
    #     [-1, tf.keras.backend.prod(recon_error_soft_priority.get_shape()[1:])]
    # )
    recon_error_soft_reduced = tf.reduce_mean(recon_error_soft, axis=[0, 2])
    note_priority_arr = tf.constant(NOTE_PRIORITY_ARRAY, dtype=recon_error.dtype)
    recon_error_soft_flat = recon_error_soft_reduced * note_priority_arr
    return tf.reduce_mean(input_tensor=recon_error_soft_flat)


if __name__ == "__main__":
    audio_path = "checkpoints/ytd_audio_00105_TRFSJUR12903CB23E7.mp3.wav"
    audio_path = "checkpoints/ytd_audio_00088_TRBHGWP128E0793AD8.mp3.wav"
    app = DrumTranscription()
    # pred = app.transcribe(audio_path)
    app.train("/host/home/76_pop_rhythm/drum_train_feature", model_name="test_drum")
