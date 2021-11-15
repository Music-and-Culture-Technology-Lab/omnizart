import pytest

from omnizart.chord import app


@pytest.mark.parametrize("mode", [None, "ChordV1"])
def test_load_model(mode):
    app._load_model(mode, custom_objects=app.custom_objects)
