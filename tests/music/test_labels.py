import pytest

from omnizart.music.labels import LabelType


def test_invalid_label_conversion_mode():
    with pytest.raises(ValueError):
        LabelType("unknown-mode")


CUSTOM_LABEL_DATA = [
    {60: {0: 1}}, {60: {0: 1}}, {60: {0: 1}}, {60: {0: 1}}, {60: {0: 1}}, {60: {0: 1}},
    {99: {41: 1}}, {99: {41: 1}}, {99: {41: 1}}, {99: {41: 1}}, {99: {41: 1}}, {99: {41: 1}}
]


@pytest.mark.parametrize("mode, expected_out_shape", [
    ("true-frame", (12, 352, 2)),
    ("frame", (12, 352, 3)),
    ("note", (12, 352, 3)),
    ("true-frame-stream", (12, 352, 12)),
    ("frame-stream", (12, 352, 23)),
    ("note-stream", (12, 352, 23)),
    ("pop-note-stream", (12, 352, 13))
])
def test_normal_label_conversion(mode, expected_out_shape):
    conv_func = LabelType(mode).get_conversion_func()
    output = conv_func(CUSTOM_LABEL_DATA)
    assert output.shape == expected_out_shape
