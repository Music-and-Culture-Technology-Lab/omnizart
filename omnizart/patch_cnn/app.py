import os
from os.path import join as jpath
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf
from mir_eval import sonify
from mir_eval.util import midi_to_hz
from scipy.io.wavfile import write as wavwrite

from omnizart.io import write_yaml, write_agg_f0_results
from omnizart.utils import get_logger, parallel_generator, get_filename, ensure_path_exists, aggregate_f0_info
from omnizart.base import BaseTranscription, BaseDatasetLoader
from omnizart.constants import datasets as d_struct
from omnizart.feature.cfp import extract_patch_cfp
from omnizart.setting_loaders import PatchCNNSettings
from omnizart.models.patch_cnn import patch_cnn_model
from omnizart.patch_cnn.inference import inference
from omnizart.train import get_train_val_feat_file_list


logger = get_logger("Patch CNN Transcription")


class PatchCNNTranscription(BaseTranscription):
    """Application class of PatchCNN module."""
    def __init__(self, conf_path=None):
        super().__init__(PatchCNNSettings, conf_path=conf_path)

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
        agg_f0: list[dict]
            List of aggregated F0 information, with each entry containing the onset, offset,
            and freqeuncy (Hz).

        See Also
        --------
        omnizart.cli.patch_cnn.transcribe: The coressponding command line entry.
        """
        if not os.path.isfile(input_audio):
            raise FileNotFoundError(f"The given audio path does not exist. Path: {input_audio}")

        logger.info("Loading model...")
        model, model_settings = self._load_model(model_path)

        logger.info("Extracting patch CFP feature...")
        feat, mapping, zzz, cenf = extract_patch_cfp(
            input_audio,
            patch_size=model_settings.feature.patch_size,
            threshold=model_settings.feature.peak_threshold,
            down_fs=model_settings.feature.sampling_rate,
            hop=model_settings.feature.hop_size,
            win_size=model_settings.feature.window_size,
            fr=model_settings.feature.frequency_resolution,
            fc=model_settings.feature.frequency_center,
            tc=model_settings.feature.time_center,
            g=model_settings.feature.gamma,
            bin_per_octave=model_settings.feature.bins_per_octave,
        )

        logger.info("Predicting...")
        feat = np.expand_dims(feat, axis=-1)
        pred = model.predict(feat)

        logger.info("Inferring contour...")
        contour = inference(
            pred=pred,
            mapping=mapping,
            zzz=zzz,
            cenf=cenf,
            threshold=model_settings.inference.threshold,
            max_method=model_settings.inference.max_method
        )
        agg_f0 = aggregate_f0_info(contour, t_unit=model_settings.feature.hop_size)

        output = self._output_midi(output, input_audio, verbose=False)
        if output is not None:
            # Output contour information
            write_agg_f0_results(agg_f0, output_path=f"{output}_f0.csv")

            # Synthesize audio
            timestamp = np.arange(len(contour)) * model_settings.feature.hop_size
            wav = sonify.pitch_contour(
                timestamp, contour, model_settings.feature.sampling_rate, amplitudes=0.5 * np.ones(len(contour))
            )
            wavwrite(f"{output}_trans.wav", model_settings.feature.sampling_rate, wav)
            logger.info("Text and Wav files have been written to %s", os.path.abspath(os.path.dirname(output)))

        return agg_f0

    def generate_feature(self, dataset_path, patch_cnn_settings=None, num_threads=4):
        """Extract the feature from the given dataset.

        To train the model, the first step is to pre-process the data into feature
        representations. After downloading the dataset, use this function to generate
        the feature by giving the path of the stored dataset.

        To specify the output path, modify the attribute
        ``patch_cnn_settings.dataset.feature_save_path``.
        It defaults to the folder of the stored dataset, and creates
        two folders: ``train_feature`` and ``test_feature``.

        Parameters
        ----------
        dataset_path: Path
            Path to the downloaded dataset.
        patch_cnn_settings: PatchCNNSettings
            The configuration instance that holds all relative settings for
            the life-cycle of building a model.
        num_threads:
            Number of threads for parallel extraction of the feature.

        See Also
        --------
        omnizart.constants.datasets:
            The supported datasets and the corresponding training/testing splits.
        """
        settings = self._validate_and_get_settings(patch_cnn_settings)

        struct = d_struct.MIR1KStructure

        ## Below are examples of dealing with multiple supported datasets.  # noqa: E266
        # dataset_type = resolve_dataset_type(
        #     dataset_path,
        #     keywords={"maps": "maps", "musicnet": "musicnet", "maestro": "maestro", "rhythm": "pop", "pop": "pop"}
        # )
        # if dataset_type is None:
        #     logger.warning(
        #         "The given path %s does not match any built-in processable dataset. Do nothing...",
        #         dataset_path
        #     )
        #     return
        # logger.info("Inferred dataset type: %s", dataset_type)
        # # Build instance mapping
        # struct = {
        #     "maps": d_struct.MapsStructure,
        #     "musicnet": d_struct.MusicNetStructure,
        #     "maestro": d_struct.MaestroStructure,
        #     "pop": d_struct.PopStructure
        # }[dataset_type]
        # label_extractor = {
        #     "maps": MapsLabelExtraction,
        #     "musicnet": MusicNetLabelExtraction,
        #     "maestro": MaestroLabelExtraction,
        #     "pop": PopLabelExtraction
        # }[dataset_type]

        # Fetching wav files
        train_data_pair = struct.get_train_data_pair(dataset_path=dataset_path)
        test_data_pair = struct.get_test_data_pair(dataset_path=dataset_path)
        logger.info("Number of total training wavs: %d", len(train_data_pair))
        logger.info("Number of total testing wavs: %d", len(test_data_pair))

        # Resolve feature output path
        train_feat_out_path, test_feat_out_path = self._resolve_feature_output_path(dataset_path, settings)
        logger.info("Output training feature to %s", train_feat_out_path)
        logger.info("Output testing feature to %s", test_feat_out_path)

        # Feature extraction
        logger.info(
            "Start extracting the training feature. "
            "This may take time to finish and affect the computer's performance"
        )
        _parallel_feature_extraction(
            train_data_pair, out_path=train_feat_out_path, feat_settings=settings.feature, num_threads=num_threads
        )
        logger.info(
            "Start extracting the testing feature. "
            "This may take time to finish and affect the computer's performance"
        )
        _parallel_feature_extraction(
            test_data_pair, out_path=test_feat_out_path, feat_settings=settings.feature, num_threads=num_threads
        )

        # Writing out the settings
        write_yaml(settings.to_json(), jpath(train_feat_out_path, ".success.yaml"))
        write_yaml(settings.to_json(), jpath(test_feat_out_path, ".success.yaml"))
        logger.info("All done")

    def train(self, feature_folder, model_name=None, input_model_path=None, patch_cnn_settings=None):
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
        patch_cnn_settings: VocalContourSettings
            The configuration that holds all relative settings for
            the life-cycle of model building.
        """
        settings = self._validate_and_get_settings(patch_cnn_settings)

        if input_model_path is not None:
            logger.info("Continue to train on model: %s", input_model_path)
            model, prev_set = self._load_model(input_model_path, custom_objects=self.custom_objects)
            settings.feature.patch_size = prev_set.feature.patch_size

        logger.info("Constructing dataset instance")
        split = settings.training.steps / (settings.training.steps + settings.training.val_steps)
        train_feat_files, val_feat_files = get_train_val_feat_file_list(feature_folder, split=split)

        output_types = (tf.float32, tf.float32)
        output_shapes = ((settings.feature.patch_size, settings.feature.patch_size, 1), (2))
        train_dataset = PatchCNNDatasetLoader(
                feature_files=train_feat_files,
                num_samples=settings.training.epoch * settings.training.batch_size * settings.training.steps
            ) \
            .get_dataset(settings.training.batch_size, output_types=output_types, output_shapes=output_shapes)
        val_dataset = PatchCNNDatasetLoader(
                feature_files=val_feat_files,
                num_samples=settings.training.epoch * settings.training.val_batch_size * settings.training.val_steps
            ) \
            .get_dataset(settings.training.val_batch_size, output_types=output_types, output_shapes=output_shapes)

        if input_model_path is None:
            logger.info("Constructing new model")
            model = patch_cnn_model(patch_size=settings.feature.patch_size)

        logger.info("Compiling model")
        optimizer = tf.keras.optimizers.Adam(learning_rate=settings.training.init_learning_rate)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

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

        logger.info("Constrcuting callbacks")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=settings.training.early_stop),
            tf.keras.callbacks.ModelCheckpoint(jpath(model_save_path, "weights.h5"), save_weights_only=True)
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


def extract_label(label_path, label_loader, mapping, cenf, t_unit):
    """Label extraction function of PatchCNN module.

    Extracts the label representation required by PatchCNN module.
    The output dimesions are: patch_length x 2. The second dimension indicates whether
    there is an active vocal pitch or not of that patch.

    Small probabilities are assigned to those patch with pitch slightly shifted
    to augment the sparse label. The probabilities are computed according to the distance
    of that pitch index to the ground-truth index: 1 / (dist + 1).

    Parameters
    ----------
    label_path: Path
        Path to the ground-truth file.
    label_loader:
        Label loader that contains ``load_label`` function for parsing the ground-truth
        file into list :class:`Label` representation.
    mapping: 2D numpy array
        The original frequency and time index of patches.
        See ``omnizart.feature.cfp.extract_patch_cfp`` for more details.
    cenf: list[float]
        Center frequencies in Hz of each frequency index.
    t_unit: float
        Time unit of each frame in seconds.

    Returns
    -------
    gt_roll: 2D numpy array
        A sequence of binary classes, represents whether the patch contains the pitch
        of vocal.
    """
    labels = label_loader.load_label(label_path)
    total_len = len(mapping)
    cenf = np.array(cenf)
    gt_roll = np.zeros((total_len, 2))
    for label in labels:
        start_tidx = int(round(label.start_time / t_unit))
        end_tidx = int(round(label.end_time / t_unit))
        frm_start = np.argmin(np.abs(mapping[:, 1] - start_tidx))
        frm_end = total_len - np.argmin(np.abs(mapping[::-1, 1] - end_tidx))
        cur_hz = midi_to_hz(label.note)
        pitch_idx = np.argmin(np.abs(cenf - cur_hz))
        for idx in range(frm_start, frm_end):
            dist = abs(mapping[idx, 0] - pitch_idx)
            prob = 1 / (1 + dist)
            gt_roll[idx, 1] = prob

    gt_roll[:, 0] = 1 - gt_roll[:, 1]
    return gt_roll


def _all_in_one_extract(data_pair, **feat_params):
    feat, mapping, zzz, cenf = extract_patch_cfp(data_pair[0], **feat_params)
    label = extract_label(data_pair[1], d_struct.MIR1KStructure, mapping=mapping, cenf=cenf, t_unit=feat_params["hop"])
    return feat, mapping, zzz, label


def _parallel_feature_extraction(data_pair_list, out_path, feat_settings, num_threads=4):
    feat_params = {
        "patch_size": feat_settings.patch_size,
        "threshold": feat_settings.peak_threshold,
        "down_fs": feat_settings.sampling_rate,
        "hop": feat_settings.hop_size,
        "win_size": feat_settings.window_size,
        "fr": feat_settings.frequency_resolution,
        "fc": feat_settings.frequency_center,
        "tc": feat_settings.time_center,
        "g": feat_settings.gamma,
        "bin_per_octave": feat_settings.bins_per_octave,
    }

    iters = enumerate(
        parallel_generator(
            _all_in_one_extract,
            data_pair_list,
            max_workers=num_threads,
            use_thread=True,
            chunk_size=num_threads,
            **feat_params
        )
    )
    for idx, ((feat, mapping, zzz, label), audio_idx) in iters:
        audio = data_pair_list[audio_idx][0]

        # logger.info("Progress: %s/%s - %s", idx+1, len(data_pair_list), audio)
        print(f"Progress: {idx + 1}/{len(data_pair_list)} - {audio}", end="\r")

        filename = get_filename(audio)
        out_hdf = jpath(out_path, filename + ".hdf")
        with h5py.File(out_hdf, "w") as out_f:
            out_f.create_dataset("feature", data=feat)
            out_f.create_dataset("mapping", data=mapping)
            out_f.create_dataset("Z", data=zzz)
            out_f.create_dataset("label", data=label)
    print("")


class PatchCNNDatasetLoader(BaseDatasetLoader):
    """Dataset loader for PatchCNN module."""
    def _get_feature(self, hdf_name, slice_start):
        feat = self.hdf_refs[hdf_name][self.feat_col_name][slice_start:slice_start + self.slice_hop].squeeze()
        return np.expand_dims(feat, axis=-1)
