import pytest

from omnizart.vocal_contour import app


@pytest.mark.parametrize("mode", [None, "VocalContour"])
def test_load_model(mode):
    app._load_model(mode, custom_objects=app.custom_objects)
