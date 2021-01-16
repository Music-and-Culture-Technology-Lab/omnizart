import os
from os.path import join as jpath
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf

from omnizart.io import write_yaml
from omnizart.base import BaseTranscription, BaseDatasetLoader
from omnizart.train import get_train_val_feat_file_list
from omnizart.utils import get_logger, ensure_path_exists, parallel_generator
from omnizart.constants.datasets import MusicNetStructure
from omnizart.setting_loaders import BeatSettings
from omnizart.beat.features import extract_musicnet_feature, extract_musicnet_label, extract_feature_from_midi
from omnizart.beat.prediction import predict
from omnizart.beat.inference import inference
from omnizart.models.rnn import blstm, blstm_attn
from omnizart.models.t2t import MultiHeadAttention


logger = get_logger("Beat Transcription")


class BeatTranscription(BaseTranscription):
    """Application class for beat tracking in MIDI domain."""
    def __init__(self, conf_path=None):
        super().__init__(BeatSettings, conf_path=conf_path)

        self.custom_objects = {"MultiHeadAttention": MultiHeadAttention}

    def transcribe(self, input_audio, model_path=None, output="./"):
        """Transcribe beat positions in the given MIDI.

        Tracks the beat in symbolic domain. Outputs three files if the output path is given:
        *<filename>.mid*, <filename>_beat.csv, and <filename>_down_beat.csv, where *filename*
        is the name of the input MIDI without extension. The *.csv files records the beat
        positions in seconds.

        Parameters
        ----------
        input_audio: Path
            Path to the MIDI file (.mid).
        model_path: Path
            Path to the trained model or the supported transcription mode.
        output: Path (optional)
            Path for writing out the transcribed MIDI file. Default to the current path.

        Returns
        -------
        midi: pretty_midi.PrettyMIDI
            The transcribed beat positions. There are two types of beat: beat and down beat.
            Each are recorded in independent instrument track.

        See Also
        --------
        omnizart.cli.beat.transcribe: CLI entry point of this function.
        """
        if not os.path.isfile(input_audio):
            raise FileNotFoundError(f"The given audio path does not exist. Path: {input_audio}")

        logger.info("Loading model...")
        model, model_settings = self._load_model(model_path, custom_objects=self.custom_objects)

        logger.info("Extracting feature...")
        feature = extract_feature_from_midi(input_audio, t_unit=model_settings.feature.time_unit)

        logger.info("Predicting...")
        pred = predict(feature, model, timesteps=model_settings.model.timesteps, batch_size=16)

        logger.info("Inferring beats and down beats...")
        midi = inference(
            pred,
            beat_th=model_settings.inference.beat_threshold,
            down_beat_th=model_settings.inference.down_beat_threshold,
            min_dist=model_settings.inference.min_distance,
            t_unit=model_settings.feature.time_unit
        )

        output = self._output_midi(output=output, input_audio=input_audio, midi=midi)
        if output is not None:
            _write_csv(midi, output=output.replace(".mid", ""))
            logger.info("MIDI and CSV file have been written to %s", output)
        return midi

    def generate_feature(self, dataset_path, beat_settings=None, num_threads=8):
        """Extract the feature from the given dataset.

        To train the model, the first step is to pre-process the data into feature
        representations. After downloading the dataset, use this function to generate
        the feature by giving the path of the stored dataset.

        To specify the output path, modify the attribute
        ``beat_settings.dataset.feature_save_path``.
        It defaults to the folder under where the dataset stored, generating
        two folders: ``train_feature`` and ``test_feature``.

        Parameters
        ----------
        dataset_path: Path
            Path to the downloaded dataset.
        beat_settings: BeatSettings
            The configuration instance that holds all relative settings for
            the life-cycle of building a model.
        num_threads:
            Number of threads for parallel extraction the feature.
        """
        settings = self._validate_and_get_settings(beat_settings)

        # Resolve feature output path
        train_feat_out_path, test_feat_out_path = self._resolve_feature_output_path(dataset_path, settings)
        logger.info("Output training feature to %s", train_feat_out_path)
        logger.info("Output testing feature to %s", test_feat_out_path)

        train_labels = MusicNetStructure.get_train_labels(dataset_path)
        test_labels = MusicNetStructure.get_test_labels(dataset_path)

        logger.info(
            "Start extract training feature of the dataset. "
            "This may take time to finish and affect the computer's performance"
        )
        _parallel_feature_extraction(train_labels, train_feat_out_path, settings.feature, num_threads=num_threads)

        logger.info(
            "Start extract test feature of the dataset. "
            "This may take time to finish and affect the computer's performance"
        )
        _parallel_feature_extraction(test_labels, test_feat_out_path, settings.feature, num_threads=num_threads)

        # Writing out the settings
        write_yaml(settings.to_json(), jpath(train_feat_out_path, ".success.yaml"))
        write_yaml(settings.to_json(), jpath(test_feat_out_path, ".success.yaml"))
        logger.info("All done")

    def train(self, feature_folder, model_name=None, input_model_path=None, beat_settings=None):
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
        beat_settings: BeatSettings
            The configuration that holds all relative settings for
            the life-cycle of model building.
        """
        settings = self._validate_and_get_settings(beat_settings)

        if input_model_path is not None:
            logger.info("Continue to train on model: %s", input_model_path)
            model, prev_set = self._load_model(input_model_path)
            settings.model.from_json(prev_set.model.to_json())
            settings.feature.time_unit = prev_set.feature.time_unit

        logger.info("Constructing dataset instance")
        split = settings.training.steps / (settings.training.steps + settings.training.val_steps)
        train_feat_files, val_feat_files = get_train_val_feat_file_list(feature_folder, split=split)

        output_types = (tf.float32, tf.float32)
        output_shapes = ((settings.model.timesteps, 178), (settings.model.timesteps, 2))
        train_dataset = BeatDatasetLoader(
                feature_files=train_feat_files,
                num_samples=settings.training.epoch * settings.training.batch_size * settings.training.steps,
                slice_hop=settings.model.timesteps // 2
            ) \
            .get_dataset(settings.training.batch_size, output_types=output_types, output_shapes=output_shapes)
        val_dataset = BeatDatasetLoader(
                feature_files=val_feat_files,
                num_samples=settings.training.epoch * settings.training.val_batch_size * settings.training.val_steps,
                slice_hop=settings.model.timesteps // 2
            ) \
            .get_dataset(settings.training.val_batch_size, output_types=output_types, output_shapes=output_shapes)

        if input_model_path is None:
            logger.info("Constructing new %s model for training.", settings.model.model_type)
            model_func = {
                "blstm": self._construct_blstm_model,
                "blstm_attn": self._construct_blstm_attn_model
            }[settings.model.model_type]
            model = model_func(settings)

        logger.info("Compiling model")
        optimizer = tf.keras.optimizers.Adam(learning_rate=settings.training.init_learning_rate)
        loss = lambda y, x: weighted_binary_crossentropy(y, x, down_beat_weight=settings.training.down_beat_weight)
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

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
            tf.keras.callbacks.EarlyStopping(
                patience=settings.training.early_stop, monitor="val_loss", restore_best_weights=False
            ),
            tf.keras.callbacks.ModelCheckpoint(
                jpath(model_save_path, "weights.h5"), save_weights_only=True, monitor="val_loss"
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

    def _construct_blstm_model(self, settings):  # pylint:disable=no-self-use
        return blstm(
            timesteps=settings.model.timesteps,
            input_dim=178,
            hidden_dim=settings.model.lstm_hidden_dim,
            num_lstm_layers=settings.model.num_lstm_layers
        )

    def _construct_blstm_attn_model(self, settings):  # pylint:disable=no-self-use
        return blstm_attn(
            timesteps=settings.model.timesteps,
            input_dim=178,
            lstm_hidden_dim=settings.model.lstm_hidden_dim,
            num_lstm_layers=settings.model.num_lstm_layers,
            attn_hidden_dim=settings.model.attn_hidden_dim
        )


def _all_in_one_extract(label_path, meter=4, t_unit=0.01):
    feature = extract_musicnet_feature(label_path, t_unit=t_unit)
    beat_arr, down_beat_arr = extract_musicnet_label(label_path, meter=meter, t_unit=t_unit)
    return feature, beat_arr, down_beat_arr


def _parallel_feature_extraction(feat_list, out_path, feat_settings, num_threads=4):
    iters = enumerate(
        parallel_generator(
            _all_in_one_extract,
            feat_list,
            max_workers=num_threads,
            chunk_size=num_threads,
            t_unit=feat_settings.time_unit
        )
    )

    for idx, ((feature, beat_arr, down_beat_arr), feat_idx) in iters:
        feat = feat_list[feat_idx]

        print(f"Progress: {idx+1}/{len(feat_list)} - {feat}" + " "*6, end="\r")  # noqa: E226
        # logger.info("Progress: %s/%s - %s", idx+1, len(feat_list), feat)

        filename, _ = os.path.splitext(os.path.basename(feat))
        out_hdf = jpath(out_path, filename + ".hdf")
        with h5py.File(out_hdf, "w") as out_f:
            out_f.create_dataset("feature", data=feature)
            out_f.create_dataset("beat", data=beat_arr)
            out_f.create_dataset("down_beat", data=down_beat_arr)
    print("")


class BeatDatasetLoader(BaseDatasetLoader):
    """Data loader for training the model of ``beat``.

    Each feature slice will have an overlap size of *timesteps//2*.

    Parameters
    ----------
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

    Yields
    ------
    feature:
        Input features for model training.
    label:
        Corresponding labels.
    """
    def _get_feature(self, hdf_name, slice_start):
        feat = self.hdf_refs[hdf_name]["feature"]
        start_idx = max(0, slice_start - self.slice_hop)
        end_idx = min(len(feat), slice_start + self.slice_hop)
        return feat[start_idx:end_idx]

    def _get_label(self, hdf_name, slice_start):
        beat = self.hdf_refs[hdf_name]["beat"]
        down_beat = self.hdf_refs[hdf_name]["down_beat"]
        start_idx = max(0, slice_start - self.slice_hop)
        end_idx = min(len(beat), slice_start + self.slice_hop)

        beat_slice = beat[start_idx:end_idx]
        db_slice = down_beat[start_idx:end_idx]
        label = np.stack([beat_slice, db_slice], axis=1)
        return label

    def _pre_yield(self, feature, label):
        feat_len = len(feature)
        label_len = len(label)
        timesteps = self.slice_hop * 2

        if (feat_len == timesteps) and (label_len == timesteps):
            # All normal
            return feature, label

        # The length of feature and label are inconsistent. Trim to the same size as the shorter one.
        if feat_len > label_len:
            feature = feature[:label_len]
            feat_len = len(feature)
        else:
            label = label[:feat_len]
            label_len = len(label)

        if feat_len != timesteps:
            assert feat_len < timesteps
            diff = timesteps - feat_len
            feature = np.pad(feature, ((0, diff), (0, 0)))
        if label_len != timesteps:
            assert label_len < timesteps
            diff = timesteps - label_len
            label = np.pad(label, ((0, diff), (0, 0)))

        return feature, label


def weighted_binary_crossentropy(target, pred, down_beat_weight=5):
    """Wrap around binary crossentropy loss with weighting to different channels."""
    # Compute binary crossentropy loss
    epsilon = 1e-6
    bce = target * tf.math.log(pred + epsilon)
    bce += (1 - target) * tf.math.log(1 - pred + epsilon)

    # Construct weight tensor
    shape_list = tf.shape(bce)
    beat_weight = tf.ones(shape_list[:2], dtype=bce.dtype)
    db_weight = tf.fill(shape_list[:2], down_beat_weight)
    db_weight = tf.cast(db_weight, bce.dtype)
    weight = tf.stack([beat_weight, db_weight], axis=-1)

    bce *= weight
    return tf.reduce_mean(-bce)


def _write_csv(midi, output):
    """Write out the beat and down beat information to files."""
    for inst in midi.instruments:
        if inst.name == "Beat":
            out_name = f"{output}_beat.csv"
        else:
            out_name = f"{output}_down_beat.csv"
        with open(out_name, "w") as out:
            onsets = [f"{nn.start:.6f}\n" for nn in inst.notes]
            out.writelines(onsets)


if __name__ == "__main__":
    app = BeatTranscription()
    out = app.transcribe("/media/whitebreeze/本機磁碟/MusicNet/test_labels/midi/2303.mid")
