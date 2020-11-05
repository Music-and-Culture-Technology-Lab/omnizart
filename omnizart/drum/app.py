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
from omnizart.drum.inference import inference
from omnizart.models.spectral_norm_net import drum_model, ConvSN2D
from omnizart.utils import get_logger, ensure_path_exists, parallel_generator, write_yaml
from omnizart.base import BaseTranscription
from omnizart.setting_loaders import DrumSettings
from omnizart.train import get_train_val_feat_file_list
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
            Path to the trained model or the supported transcription mode.
        output: Path (optional)
            Path for writing out the transcribed MIDI file. Default to the current path.

        Returns
        -------
        midi: pretty_midi.PrettyMIDI
            The transcribed drum notes.

        See Also
        --------
        omnizart.cli.drum.transcribe: CLI entry point of this function.
        """
        if not os.path.isfile(input_audio):
            raise FileNotFoundError(f"The given audio path does not exist. Path: {input_audio}")

        # Extract feature according to model configuration
        logger.info("Extracting feature...")
        patch_cqt_feature, mini_beat_arr = extract_patch_cqt(input_audio)

        # Load model configurations
        logger.info("Loading model...")
        model, model_settings = self._load_model(model_path, custom_objects=self.custom_objects)

        logger.info("Predicting...")
        pred = predict(patch_cqt_feature, model, model_settings.feature.mini_beat_per_segment)
        logger.debug("Prediction shape: %s", pred.shape)

        logger.info("Infering MIDI...")
        midi = inference(
            pred,
            mini_beat_arr,
            bass_drum_th=model_settings.inference.bass_drum_th,
            snare_th=model_settings.inference.snare_th,
            hihat_th=model_settings.inference.hihat_th
        )

        if output is not None:
            save_to = output
            if os.path.isdir(save_to):
                save_to = jpath(save_to, os.path.basename(input_audio).replace(".wav", ".mid"))
            midi.write(save_to)
            logger.info("MIDI file have been written to %s", save_to)

        logger.info("Transcription finished")
        return midi

    def generate_feature(self, dataset_path, drum_settings=None, num_threads=3):
        """Extract the feature of the whole dataset.

        Currently only supports Pop dataset. To train the model, you have to prepare the training
        data first, then process it into feature representations. After downloading the dataset,
        use this function to do the pre-processing and transform the raw data into features.

        To specify the output path, modify the attribute
        ``music_settings.dataset.feature_save_path`` to the value you want.
        It will default to the folder under where the dataset stored, generating
        two folders: ``train_feature`` and ``test_feature``.

        Parameters
        ----------
        dataset_path: Path
            Path to the downloaded dataset.
        drum_settings: DrumSettings
            The configuration instance that holds all relative settings for
            the life-cycle of building a model.
        num_threads:
            Number of threads for parallel extracting the features.

        See Also
        --------
        omnizart.constants.datasets.PopStructure:
            The only supported dataset for drum transcription. Records the train/test
            partition according to the folder.
        """
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
        """Model training.

        Train a new model or continue to train on a previously trained model.

        Parameters
        ----------
        feature_folder: Path
            Path to the folder containing generated feature.
        model_name: str
            The name for storing the trained model. If not given, will default to the
            current timesamp.
        input_model_path: Path
            Continue to train on the pre-trained model by specifying the path.
        drum_settings: DrumSettings
            The configuration instance that holds all relative settings for
            the life-cycle of building a model.
        """
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
            epochs=settings.training.epoch,
            batch_size=settings.training.batch_size,
            steps=settings.training.steps
        )
        val_dataset = get_dataset(
            feature_files=val_feat_files,
            epochs=settings.training.epoch,
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
        model.compile(optimizer=optimizer, loss=loss_func, metrics=["accuracy"])

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
            tf.keras.callbacks.EarlyStopping(patience=settings.training.early_stop, monitor="val_loss"),
            tf.keras.callbacks.ModelCheckpoint(jpath(model_save_path, "weights.h5"), save_weights_only=True)
        ]
        logger.info("Callback list: %s", callbacks)

        logger.info("Start training")
        history = model.fit(
            train_dataset,
            validate_data=val_dataset,
            epochs=settings.training.epoch,
            steps_per_epoch=settings.training.steps,
            validation_steps=settings.training.val_steps,
            callbacks=callbacks,
            use_multiprocessing=True,
            workers=8
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
                    saved = True
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


def loss_func(target, pred, soft_loss_range=20):
    recon_error = tf.abs(target*100 - pred)  # noqa: E226
    recon_error_soft = tf.where(
        recon_error <= soft_loss_range,
        tf.zeros_like(recon_error),
        recon_error - soft_loss_range
    )

    recon_error_soft_reduced = tf.reduce_mean(recon_error_soft, axis=[0, 2])
    note_priority_arr = tf.constant(NOTE_PRIORITY_ARRAY, dtype=recon_error.dtype)
    recon_error_soft_flat = recon_error_soft_reduced * note_priority_arr
    return tf.reduce_mean(input_tensor=recon_error_soft_flat)
