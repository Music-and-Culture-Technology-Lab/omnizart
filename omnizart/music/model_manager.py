"""Process configurations of the model

Records the relative settings while training the model, and keep it for future inference.
Also use the `ModelInfo` class for creating the new model for training.
"""

# pylint: disable=R1705,E0611

import os
import json

import h5py
import numpy as np
from scipy.special import expit
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Conv2D

from omnizart.models.t2t import local_attention_2d, split_heads_2d, combine_heads_2d
from omnizart.music.utils import create_batches, cut_batch_pred, cut_frame
from omnizart.models.u_net import semantic_segmentation, semantic_segmentation_attn, multihead_attention
from omnizart.constants.feature import HARMONIC_NUM


class ModelManager:
    """Manages model-relevant utilities.

    Create and load model, save and load model configurations.
    """
    def __init__(self, model_name="MyModel"):
        self.name = model_name
        self.output_classes = None
        self.label_type = None
        self.timesteps = 256
        self.feature_type = "CFP"
        self.input_channels = [1, 3]
        self.frm_th = 0.5
        self.inst_th = 1.1
        self.onset_th = 6
        self.dura_th = 0
        self.description = None
        self.model_type = None

        # Trainig information
        self.dataset = None
        self.epochs = None
        self.steps = None
        self.train_batch_size = None
        self.val_batch_size = None
        self.early_stop = None
        self.loss_function = None

        # Other inner settings
        self._num_gpus = 2
        self._custom_layers = {
            "multihead_attention": multihead_attention,
            "Conv2D": Conv2D,
            "split_heads_2d": split_heads_2d,
            "local_attention_2d": local_attention_2d,
            "combine_heads_2d": combine_heads_2d,
        }

    def create_model(self, model_type="attn"):
        self._validate_args()
        self.model_type = model_type
        if model_type == "aspp":
            return semantic_segmentation(
                feature_num=384,
                ch_num=len(self.input_channels),
                timesteps=self.timesteps,
                out_class=self.output_classes,
                multi_grid_layer_n=1,
                multi_grid_n=3,
            )
        elif model_type == "attn":
            return semantic_segmentation_attn(
                feature_num=384,
                ch_num=len(self.input_channels),
                timesteps=self.timesteps,
                out_class=self.output_classes,
            )
        else:
            raise ValueError(f"Invalid mode: {model_type}. Available: ['attn', 'aspp']")

    def _validate_args(self):
        assert self.output_classes is not None
        assert self.label_type is not None

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The given path doesn't exist: {model_path}.")

        # Load model architecture
        self.name = os.path.basename(model_path)
        model = model_from_yaml(open(os.path.join(model_path, "arch.yaml")).read(), custom_objects=self._custom_layers)

        # Load weights of the model
        weight_path = os.path.join(model_path, "weights.h5")
        with h5py.File(weight_path, "r") as weight:
            keys = list(weight.keys())
            is_para = any(["model" in k for k in keys])

        if is_para:
            para_model = multi_gpu_model(model, gpus=self._num_gpus)
            para_model.load_weights(weight_path)
            model = para_model.layers[-2]
        else:
            model.load_weights(weight_path)

        # Load related configurations
        conf = json.load(open(os.path.join(model_path, "configuration.json")))
        self.name = conf.get("model_name", self.name)
        self.output_classes = conf.get("output_classes", self.output_classes)
        self.label_type = conf.get("label_type", self.label_type)
        self.timesteps = conf.get("timesteps", self.timesteps)
        self.frm_th = conf.get("frame_threshold", self.frm_th)
        self.inst_th = conf.get("instrument_threshold", self.inst_th)
        self.onset_th = conf.get("onset_threshold", self.onset_th)
        self.dura_th = conf.get("duration_threshold", self.dura_th)
        self.description = conf.get("description", self.description)
        self.feature_type = conf.get("feature_type", self.feature_type)
        self.input_channels = conf.get("input_channels", self.input_channels)
        self.dataset = conf.get("training_settings").get("dataset", None)

        print(f"Model {model_path} loaded")
        return model

    def save_model(self, model, save_path):
        path = os.path.join(save_path, self.name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Save model architecture/weights
        open(os.path.join(path, "arch.yaml"), "w").write(model.to_yaml())
        model.save_weights(os.path.join(path, "weights.h5"))

        # Save related configurations
        self.save_configuration(path)
        print(f"Model saved to {save_path}/{self.name}.")

    def save_configuration(self, save_path):
        conf = {
            "model_name": self.name,
            "model_type": self.model_type,
            "output_classes": self.output_classes,
            "label_type": self.label_type,
            "timesteps": self.timesteps,
            "frame_threshold": self.frm_th,
            "instrument_threshold": self.inst_th,
            "onset_threshold": self.onset_th,
            "duration_threshold": self.dura_th,
            "feature_type": self.feature_type,
            "input_channels": self.input_channels,
            "training_settings": {
                "dataset": self.dataset,
                "epochs": self.epochs,
                "steps": self.steps,
                "train_batch_size": self.train_batch_size,
                "val_batch_size": self.val_batch_size,
                "loss_function": self.loss_function,
                "early_stop": self.early_stop,
            },
            "description": self._construct_description(),
        }
        json.dump(conf, open(os.path.join(save_path, "configuration.json"), "w"), indent=2)

    def predict(self, feature, model, feature_num=384, batch_size=4):
        """Make predictions on the feature.

        Generate predictions by using the loaded model.

        Parameters
        ----------
        feature : numpy.ndarray
            Extracted feature of the audio.
            Dimension:  timesteps x feature_size x channels
        model : keras.model
            The loaded model instance
        feature_num : int
            Padding along the feature dimension to the size `feature_num`
        batch_size : int
            Batch size for each step of prediction. The size is depending on the available GPU memory.

        Returns
        -------
        pred : numpy.ndarray
            The predicted results. The values are ranging from 0~1.
        """

        # Create batches of the feature
        features = create_batches(feature, b_size=batch_size, timesteps=self.timesteps, feature_num=feature_num)

        # Container for the batch prediction
        pred = []

        # Initiate lamda function for later processing of prediction
        cut_frm = lambda x: cut_frame(x, ori_feature_size=352, feature_num=features[0][0].shape[1])

        t_len = len(features[0][0])
        first_split_start = round(t_len * 0.75)
        second_split_start = t_len + round(t_len * 0.25)

        total_batches = len(features)
        features.insert(0, [np.zeros_like(features[0][0])])
        features.append([np.zeros_like(features[0][0])])
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

    def _construct_description(self):
        return f"""Information about this model
            Model name: {self.name}
            Input feature type: {self.feature_type}
            Input channels: {self.input_channels}
            Timesteps: {self.timesteps}
            Label type: {self.label_type}
            Thresholds:
                Instrument: {self.inst_th}
                Frame: {self.frm_th}
                Onset: {self.onset_th}
                Duration: {self.dura_th}
            Training settings:
                Previously trained on {self.dataset}
                Maximum epochs: {self.epochs}
                Steps per epoch: {self.steps}
                Training batch size: {self.train_batch_size}
                Validation batch size: {self.val_batch_size}
                Loss function type: {self.loss_function}
                Early stopping: {self.early_stop}
        """

    def __repr__(self):
        return self._construct_description()

    def __str__(self):
        if self.description is None:
            self.description = self._construct_description()
        return self.description

    @property
    def feature_type(self):
        return self._feature_type

    @feature_type.setter
    def feature_type(self, f_type):
        available = ["CFP", "HCFP"]
        if f_type not in available:
            raise ValueError(f"Invalid feature type: {f_type}. Available: {available}.")
        self._feature_type = f_type

    @property
    def input_channels(self):
        return self._input_channels

    @input_channels.setter
    def input_channels(self, value):
        if not isinstance(value, list):
            value = list(value)

        if len(value) < 5:
            if not all(v in [0, 1, 2, 3] for v in value):
                raise ValueError(f"Invalid channel numbers: {value}. Available: [0, 1, 2, 3].")
        else:
            if len(value) % (HARMONIC_NUM+1) != 0:  # noqa: E226
                raise ValueError(
                    f"Invalid channel number of harmonic feature: {value}. Length should be multiple of \
                      {HARMONIC_NUM+1}."
                )

        self._input_channels = value
