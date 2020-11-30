# pylint: disable=W0201
import os
import abc

import six
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils

from omnizart.io import write_yaml
from omnizart.utils import get_logger, ensure_path_exists

logger = get_logger("Callbacks")


class Callback(metaclass=abc.ABCMeta):
    """Base class of all callback classes"""
    def __init__(self, monitor=None):
        if monitor is not None:
            self.monitor = monitor
            if "acc" in monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_train_begin(self, history=None):
        pass

    def on_train_end(self, history=None):
        pass

    def on_epoch_begin(self, epoch, history=None):
        pass

    def on_epoch_end(self, epoch, history=None):
        pass

    def on_train_batch_begin(self, history=None):
        pass

    def on_train_batch_end(self, history=None):
        pass

    def on_test_batch_begin(self, history=None):
        pass

    def on_test_batch_end(self, history=None):
        pass

    def _set_model(self, model):
        self.model = model

    def _get_monitor_value(self, history, callback_name="Callback"):
        history = history or {"train": [], "validate": []}

        if self.monitor.startswith("val"):
            hist = history["validate"]
        else:
            hist = history["train"]

        if len(hist) > 0:
            current = hist[-1]

        metric = self.monitor.split("_")[-1]
        if metric == "acc":
            metric = "accuracy"
        score = current.get(metric)
        if score is None:
            logger.warning(
                "%s conditioned on metric %s "
                "which is not available. Available metrics are %s",
                callback_name, self.monitor, list(current.keys())
            )
        return score


class EarlyStopping(Callback):
    """Early stop the training after no improvement on the monitor for a certain period.

    Parameters
    ----------
    patience: int
        Longeset period of epochs for waiting the target metrics showing improvement.
    monitor: str
        Metric name for the observation.
    """
    def __init__(self, patience=5, monitor="val_acc"):
        super().__init__(monitor=monitor)
        self.patience = patience
        self.stopped_epoch = 0

    def on_train_begin(self, history=None):
        self.wait = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, history=None):
        assert hasattr(self, "model")
        score = self._get_monitor_value(history, callback_name="Early stopping")
        if score is None:
            return

        if self.monitor_op(score, self.best):
            self.best = score
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.model.stop_training = True
            self.stopped_epoch = epoch

    def on_train_end(self, history=None):
        if self.stopped_epoch > 0:
            print("Early stopped training")


class ModelCheckpoint(Callback):
    """Saving the model during training.

    The newest checkpoint will override the original checkpoint during a single
    training period.

    Parameters
    ----------
    filepath: Path
        Path for saving the checkpoint.
    monitor: str
        Metric name for the observation. No effect if `save_bset_only` is set to false.
    save_best_only: bool
        Whether to save the model having the best performance on the metric only.
    save_weights_only: bool
        Save the model's weight only, without architecture.
    """
    def __init__(self, filepath, monitor='val_acc', save_best_only=False, save_weights_only=False):
        super().__init__(monitor=monitor)
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only

    def on_train_begin(self, history=None):
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, history=None):
        if self.save_best_only:
            score = self._get_monitor_value(history, callback_name="Model checkpoint")
            if score is None:
                return

            if self.monitor_op(score, self.best):
                self.best = score
                self._save_model()
        else:
            self._save_model()

    def _ensure_path_exists(self):
        if hasattr(self, "_path_checked") and self._path_checked:  # pylint: disable=E0203
            return
        ensure_path_exists(self.filepath)
        self._path_checked = True

    def _save_model(self):
        self._ensure_path_exists()
        if not self.save_weights_only:
            write_yaml(self.model.to_yaml(), os.path.join(self.filepath, "arch.yaml"), dump=False)
        self.model.save_weights(os.path.join(self.filepath, "weights.h5"))


class TFModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """Re-implementation of Tensorflow ModelCheckpoint.

    Customize the behaviour of saving the checkpoints.
    When specify save_weights_only to 'True', save the weights only during training, and save
    the whole model including architecture using model.save() at the end of training.

    This callback is mainly designed for saving customized models that is unable to
    use model.to_yaml() function.
    """
    def set_model(self, model):
        self.model = model
        self._saved = False

        # ## Below are the original implementation, which will do some checks
        # ## and implicitly turn on the 'save_weights_only' flag, leading to
        # ## an unexpected situation that there is supposed to be a 'saved_model.pb'
        # ## in the checkpoint path, but actually not and making us unable to
        # ## use tf.keras.models.load_model to load the model.
        # >>> if (not self.save_weights_only and
        #            not model._is_graph_network and
        #            model.__class__.__name__ != 'Sequential'):
        #        self.save_weights_only = True

    def on_train_end(self, logs):
        if self.save_weights_only:
            filepath = self._get_file_path(0, logs)
            basename = os.path.basename(filepath)
            dirname = os.path.dirname(filepath)
            self.model.save(os.path.dirname(filepath), include_optimizer=False)
            remove_list = ["checkpoint", f"{basename}.data-00000-of-00001", f"{basename}.index"]
            for remove_item in remove_list:
                os.remove(os.path.join(dirname, remove_item))

    def _save_model(self, epoch, logs):
        """Saves the model.

        Parameters
        ----------
        epoch: The epoch this iteration is in.
        logs: The `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        # pylint: disable=too-many-nested-blocks,too-many-branches
        logs = logs or {}

        if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = tf_utils.to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            try:
                if not self._saved and self.save_weights_only:
                    self.model.save(os.path.dirname(filepath), overwrite=True, include_optimizer=False)
                    self._saved = True
                elif self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logger.warning('Can save best model only with %s available, skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f, \
                                    saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath))

                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True, options=self._options)
                            else:
                                self.model.save(filepath, overwrite=True, options=self._options)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                    (epoch + 1, self.monitor, self.best))  # noqa: E128
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True, options=self._options)
                    else:
                        self.model.save(filepath, overwrite=True, options=self._options)

                self._maybe_remove_file()
            except IOError as e:
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in six.ensure_str(e.args[0]).lower():
                    raise IOError(f'Please specify a non-directory filepath for ModelCheckpoint. \
                        Filepath used is an existing directory: {filepath}')
