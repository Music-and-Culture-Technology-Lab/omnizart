import pytest

from omnizart.base import Label
from omnizart.music.labels import (
    MaestroLabelExtraction,
    MapsLabelExtraction,
    MusicNetLabelExtraction,
    LabelType
)


def _assert_all_equal(expected, labels):
    assert all(label == expected_label for label, expected_label in zip(labels, expected))


def test_maestro_extract_label():
    expected = [
        Label(31.979545, 32.340909, 60, 0, 80),
        Label(32.4, 32.5, 65, 0, 80),
        Label(32.520455, 33.079545, 67, 0, 80),
        Label(33.059091, 33.479545, 65, 0, 80),
        Label(33.579545, 33.940909, 63, 0, 80)
    ]
    gt_file_path = "./tests/resource/gt_files/maestro_gt_file.mid"
    labels = MaestroLabelExtraction.load_label(gt_file_path)
    _assert_all_equal(expected, labels)


def test_maps_extract_label():
    expected = [
        Label(0.5, 1.74117, 69),
        Label(0.843864, 1.13805, 66),
        Label(0.843864, 1.13805, 50),
        Label(0.990956, 1.74117, 73),
        Label(1.13805, 1.74117, 74)
    ]
    gt_file_path = "./tests/resource/gt_files/maps_gt_file.txt"
    labels = MapsLabelExtraction.load_label(gt_file_path)
    _assert_all_equal(expected, labels)


def test_musicnet_extract_label():
    expected = [
        Label(0.231428, 0.962857, 61, instrument=41, start_beat=0, end_beat=1.489583, note_value="Dotted Half"),
        Label(0.231428, 0.568117, 65, instrument=6, start_beat=0, end_beat=0.489583, note_value="Quarter"),
        Label(0.231428, 0.510068, 46, instrument=0, start_beat=0, end_beat=0.333333, note_value="Dotted Eighth"),
        Label(0.579727, 0.660997, 63, instrument=40, start_beat=0.5, end_beat=0.739583, note_value="Eighth"),
        Label(0.579727, 0.695827, 58, instrument=70, start_beat=0.5, end_beat=0.833333, note_value="Dotted Eighth"),
        Label(0.672607, 0.777097, 65, instrument=71, start_beat=0.75, end_beat=0.989583, note_value="Eighth"),
        Label(0.777097, 1.473696, 58, instrument=73, start_beat=1.0, end_beat=2.989583, note_value="Whole"),
        Label(0.777097, 0.904807, 66, instrument=68, start_beat=1.0, end_beat=1.333333, note_value="Dotted Eighth"),
        Label(0.962857, 1.055714, 63, instrument=0, start_beat=1.5, end_beat=1.739583, note_value="Eighth"),
        Label(0.962857, 1.090566, 39, instrument=43, start_beat=1.5, end_beat=1.833333, note_value="Dotted Eighth")
    ]
    gt_file_path = "./tests/resource/gt_files/musicnet_gt_file.csv"
    labels = MusicNetLabelExtraction.load_label(gt_file_path)
    _assert_all_equal(expected, labels)


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
