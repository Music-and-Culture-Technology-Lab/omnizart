import pytest

from omnizart.drum import app


@pytest.mark.parametrize("mode", [None, "Keras"])
def test_load_model(mode):
    app._load_model(mode, custom_objects=app.custom_objects)
