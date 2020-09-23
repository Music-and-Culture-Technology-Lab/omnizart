import os
import abc

import numpy as np

from omnizart.utils import get_logger

logger = get_logger("Callbacks")


class Callback(metaclass=abc.ABCMeta):
    def __init__(self, monitor=None):
        if monitor is not None:
            self.monitor = monitor
            if "acc" in monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_train_batch_begin(self):
        pass

    def on_train_batch_end(self):
        pass

    def on_test_batch_begin(self):
        pass

    def on_test_batch_end(self):
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
                f"{callback_name} conditioned on metric %s "
                "which is not available. Available metrics are %s",
                self.monitor, list(current.keys())
            )
        return score


class EarlyStopping(Callback):
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
        if hasattr(self, "_path_checked") and self._path_checked:
            return
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
            self._path_checked = True

    def _save_model(self):
        self._ensure_path_exists()
        if not self.save_weights_only:
            with open(os.path.join(self.filepath, "arch.yaml"), "w") as out:
                out.write(self.model.to_yaml())
        self.model.save_weights(os.path.join(self.filepath, "weights.h5"))

