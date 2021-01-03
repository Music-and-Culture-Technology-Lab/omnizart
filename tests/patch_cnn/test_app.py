import pytest

from omnizart.patch_cnn import app


@pytest.mark.parametrize("mode", [None, "Melody"])
def test_load_model(mode):
    app._load_model(mode)
