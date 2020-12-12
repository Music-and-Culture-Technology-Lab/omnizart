import os

import pytest
import tensorflow as tf

from omnizart import MODULE_PATH
from omnizart.vocal import app


@pytest.mark.parametrize("mode", [None, "Semi"])
def test_load_model(mode):
    app._load_model(mode)


@pytest.mark.parametrize("mode", ["Semi"])
def test_load_pb_model(mode):
    default_path = app.settings.checkpoint_path[mode]
    model_path = os.path.join(MODULE_PATH, default_path)
    _, weight_path, _ = app._resolve_model_path(model_path)
    model = tf.keras.models.load_model(model_path)
    model.load_weights(weight_path.replace(".h5", ""))
