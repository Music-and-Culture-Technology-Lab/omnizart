import os

import pytest
import tensorflow as tf

from omnizart import MODULE_PATH
from omnizart.vocal import app


@pytest.mark.parametrize("mode", [None, "Semi"])
def test_load_model(mode):
    app._load_model(mode)

