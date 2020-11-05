import os
from os.path import join as jpath
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf

from omnizart.base import BaseTranscription
from omnizart.setting_loaders import ChordSettings
from omnizart.utils import get_logger, ensure_path_exists, parallel_generator, write_yaml
from omnizart.constants.datasets import McGillBillBoard
from omnizart.feature.chroma import extract_chroma
from omnizart.chord.features import get_train_test_split_ids, extract_feature_label
from omnizart.chord.dataset import get_dataset
from omnizart.chord.inference import inference, write_csv
from omnizart.train import get_train_val_feat_file_list
from omnizart.models.chord_model import ChordModel, ReduceSlope


logger = get_logger("Chord Application")


class ChordTranscription(BaseTranscription):
    """Application class for chord transcription."""
    def __init__(self, conf_path=None):
        super().__init__(ChordSettings, conf_path=conf_path)

    def transcribe(self, input_audio, model_path=None, output="./"):
        """Transcribe chords in the audio.

        This function transcribes chord progression in the audio and will outputs MIDI
        and CSV files. The MIDI file is provided for quick validation by directly listen
        to the chords. The complete transcription results are listed in the CSV file,
        which contains the chord's name and the start and end time.

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
            Transcribed chord progression with default chord-to-notes mappings.

        See Also
        --------
        omnizart.cli.chord.transcribe: CLI entry point of this function.
        omnizart.chord.inference: Records the default chord-to-notes mappings.
        """

        logger.info("Extracting feature")
        t_unit, chroma = extract_chroma(input_audio)

        logger.info("Loading model")
        model, settings = self._load_model(model_path)

        logger.info("Preparing feature for model prediction")
        pad_size = settings.feature.segment_width // 2
        chroma_pad = np.pad(chroma, ((pad_size, pad_size), (0, 0)), constant_values=0)
        segments = np.array([
            chroma_pad[i-pad_size:i+pad_size+1] for i in range(pad_size, pad_size+len(chroma))  # noqa: E226
        ])
        segments = segments.reshape([-1, chroma.shape[1] * settings.feature.segment_width])

        num_steps = settings.feature.num_steps
        pad_end = num_steps - len(segments) % num_steps
        segments_pad = np.pad(segments, ((0, pad_end), (0, 0)), constant_values=0)

        num_seqs = len(segments_pad) // num_steps
        segments_pad = segments_pad.reshape([num_seqs, num_steps, segments_pad.shape[1]])

        logger.info("Predicting...")
        chord, _, _, _ = model.predict(segments_pad)
        chord = chord.reshape(np.prod(chord.shape))[:-pad_end]  # Reshape and remove padding

        logger.info("Infering chords...")
        midi, info = inference(chord, t_unit)

        if output is not None:
            save_to = output
            if os.path.isdir(save_to):
                save_to = jpath(save_to, os.path.basename(input_audio).replace(".wav", ".mid"))
            midi.write(save_to)
            write_csv(info, output=save_to.replace(".mid", ".csv"))
            logger.info("MIDI and CSV file have been written to %s", save_to)

        logger.info("Transcription finished")
        return midi

    def generate_feature(self, dataset_path, chord_settings=None, num_threads=4):
        """Extract feature of McGill BillBoard dataset.

        There are three main features that will be used in the training:

        * chroma: input feature of the NN model
        * chord: the first type of the ground-truth
        * chord_change: the second type of the ground-truth

        The last two feature will be both used for computing the training loss.
        During the feature extraction, the feature data is stored as a numpy array
        with named field, makes it works like a dict type.
        """
        if chord_settings is not None:
            assert isinstance(chord_settings, ChordSettings)
            settings = chord_settings
        else:
            settings = self.settings

        index_file_path = jpath(dataset_path, McGillBillBoard.index_file_path)
        train_ids, test_ids = get_train_test_split_ids(
            index_file_path, train_test_split_id=McGillBillBoard.train_test_split_id
        )

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

        feat_path = jpath(dataset_path, McGillBillBoard.feature_folder)
        label_path = jpath(dataset_path, McGillBillBoard.label_folder)
        input_list = []
        for f_name in os.listdir(feat_path):
            if not (f_name in train_ids or f_name in test_ids):
                continue
            input_list.append((
                jpath(feat_path, f_name, McGillBillBoard.feature_file_name),
                jpath(label_path, f_name, McGillBillBoard.label_file_name)
            ))

        iters = enumerate(
            parallel_generator(
                _extract_feature_arg_wrapper,
                input_list,
                segment_width=settings.feature.segment_width,
                segment_hop=settings.feature.segment_hop,
                num_steps=settings.feature.num_steps,
                max_workers=num_threads,
                chunk_size=num_threads
            )
        )
        for idx, ((feature), feat_idx) in iters:
            feat_path = input_list[feat_idx][0]
            f_name = os.path.basename(os.path.dirname(feat_path))
            logger.info("Progress: %s/%s - %s", idx + 1, len(input_list), f_name)

            feat_out_path = train_feat_out_path if f_name in train_ids else test_feat_out_path
            out_path = jpath(feat_out_path, f_name + ".hdf")
            _write_feature(feature, out_path=out_path)

        # Writing out the settings
        write_yaml(settings.to_json(), jpath(train_feat_out_path, ".success.yaml"))
        write_yaml(settings.to_json(), jpath(test_feat_out_path, ".success.yaml"))
        logger.info("All done")

    def train(self, feature_folder, model_name=None, input_model_path=None, chord_settings=None):
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
        chord_settings: ChordSettings
            The configuration instance that holds all relative settings for
            the life-cycle of building a model.
        """

        if chord_settings is not None:
            assert isinstance(chord_settings, ChordSettings)
            settings = chord_settings
        else:
            settings = self.settings

        if input_model_path is not None:
            logger.info("Continue to train one model: %s", input_model_path)
            model, _ = self._load_model(input_model_path)

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
            model = ChordModel(
                out_classes=26,
                num_enc_attn_blocks=settings.model.num_enc_attn_blocks,
                num_dec_attn_blocks=settings.model.num_dec_attn_blocks,
                segment_width=settings.feature.segment_width,
                n_steps=settings.feature.num_steps,
                freq_size=settings.model.freq_size,
                enc_input_emb_size=settings.model.enc_input_emb_size,
                dec_input_emb_size=settings.model.dec_input_emb_size,
                dropout_rate=settings.model.dropout_rate,
                annealing_rate=settings.model.annealing_rate
            )

        learninig_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            settings.training.init_learning_rate,
            decay_steps=settings.training.steps,
            decay_rate=settings.training.learning_rate_decay,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learninig_rate, clipvalue=1)
        model.compile(optimizer=optimizer, loss=chord_loss_func, metrics=["accuracy"])

        logger.info("Resolving model output path")
        if model_name is None:
            model_name = str(datetime.now()).replace(" ", "_")
        if not model_name.startswith(settings.model.save_prefix):
            model_name = settings.model.save_prefix + "_" + model_name
            model_save_path = jpath(settings.model.save_path, model_name)
        ensure_path_exists(model_save_path)
        write_yaml(settings.to_json(), jpath(model_save_path, "configurations.yaml"))
        logger.info("Model output to: %s", model_save_path)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=settings.training.early_stop, monitor="val_loss"),
            tf.keras.callbacks.ModelCheckpoint(
                jpath(model_save_path, "weights"), save_weights_only=True, monitor="val_loss"
            ),
            ReduceSlope()
        ]

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=settings.training.epoch,
            steps_per_epoch=settings.training.steps,
            validation_steps=settings.training.val_steps,
            callbacks=callbacks
        )
        return history

    def _load_model(self, model_path=None, custom_objects=None):
        _, weight_path, conf_path = self._resolve_model_path(model_path)
        weight_path = weight_path.replace(".h5", "")
        settings = self.setting_class(conf_path=conf_path)
        model = ChordModel(
            num_enc_attn_blocks=settings.model.num_enc_attn_blocks,
            num_dec_attn_blocks=settings.model.num_dec_attn_blocks,
            segment_width=settings.feature.segment_width,
            n_steps=settings.feature.num_steps,
            freq_size=settings.model.freq_size,
            enc_input_emb_size=settings.model.enc_input_emb_size,
            dec_input_emb_size=settings.model.dec_input_emb_size,
            dropout_rate=settings.model.dropout_rate,
            annealing_rate=settings.model.annealing_rate
        )

        try:
            model.load_weights(weight_path).expect_partial()
        except tf.python.framework.errors_impl.OpError:
            raise FileNotFoundError(
                f"Weight file not found: {weight_path}. Perhaps not yet downloaded?\n"
                "Try execute 'omnizart download-checkpoints'"
            )

        return model, settings


def _extract_feature_arg_wrapper(input_tup, **kwargs):
    return extract_feature_label(input_tup[0], input_tup[1], **kwargs)


def _write_feature(feature, out_path):
    key_list = ["chroma", "chord", "chord_change", "tc", "sequence_len"]
    with h5py.File(out_path, "w") as out_hdf:
        for key in key_list:
            data = np.concatenate([feat[key] for feat in feature])
            out_hdf.create_dataset(key, data=data, compression="gzip", compression_opts=3)
        out_hdf.create_dataset("num_sequence", data=feature[0]["num_sequence"])


def chord_loss_func(
    chord,
    chord_change,
    logits,
    chord_change_logits,
    out_classes=26,
    lambda_loss_c=1,
    lambda_loss_ct=3
):
    exp_cc_logits = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(chord_change, tf.float32), logits=chord_change_logits
    )
    loss_ct = lambda_loss_ct * tf.reduce_mean(input_tensor=exp_cc_logits)

    one_hot_chord = tf.one_hot(chord, depth=out_classes)  # pylint: disable=E1120
    loss_c = lambda_loss_c * tf.compat.v1.losses.softmax_cross_entropy(
        onehot_labels=one_hot_chord, logits=logits, label_smoothing=0.1
    )

    return loss_ct + loss_c
