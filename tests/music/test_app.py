import pytest

from omnizart.music import app


@pytest.mark.parametrize("mode", [None, "Piano", "Stream", "Pop"])
def test_load_model(mode):
    app._load_model(mode, custom_objects=app.custom_objects)
