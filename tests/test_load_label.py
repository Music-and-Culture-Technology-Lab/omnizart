from omnizart.base import Label
from omnizart.constants import datasets as dset


def _assert_all_equal(expected, labels):
    assert all(label == expected_label for label, expected_label in zip(labels, expected))


def test_maestro_load_label():
    expected = [
        Label(31.979545, 32.340909, 60, 0, 80),
        Label(32.4, 32.5, 65, 0, 80),
        Label(32.520455, 33.079545, 67, 0, 80),
        Label(33.059091, 33.479545, 65, 0, 80),
        Label(33.579545, 33.940909, 63, 0, 80)
    ]
    gt_file_path = "./tests/resource/gt_files/maestro_gt_file.mid"
    labels = dset.MaestroStructure.load_label(gt_file_path)
    _assert_all_equal(expected, labels)


def test_maps_load_label():
    expected = [
        Label(0.5, 1.74117, 69),
        Label(0.843864, 1.13805, 66),
        Label(0.843864, 1.13805, 50),
        Label(0.990956, 1.74117, 73),
        Label(1.13805, 1.74117, 74)
    ]
    gt_file_path = "./tests/resource/gt_files/maps_gt_file.txt"
    labels = dset.MapsStructure.load_label(gt_file_path)
    _assert_all_equal(expected, labels)


def test_musicnet_load_label():
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
    labels = dset.MusicNetStructure.load_label(gt_file_path)
    _assert_all_equal(expected, labels)


def test_mir1k_load_label():
    expected = [
        Label(0.04, 0.06, 57.9108),
        Label(0.06, 0.08, 57.4161),
        Label(0.08, 0.1, 57.174),
        Label(0.1, 0.12, 59.7627),
        Label(0.12, 0.14, 60.0442),
        Label(0.14, 0.16, 60.3304)
    ]
    gt_file_path = "./tests/resource/gt_files/mir1k_gt_file.pv"
    labels = dset.MIR1KStructure.load_label(gt_file_path)
    _assert_all_equal(expected, labels)


def test_tonas_load_label():
    expected = [
        Label(0.233333, 0.366666, 53),
        Label(0.4, 0.583333, 58),
        Label(0.6, 0.75, 58),
        Label(0.766666, 0.95, 58)
    ]
    gt_file_path = "./tests/resource/gt_files/tonas_gt_file.notes.Corrected"
    labels = dset.TonasStructure.load_label(gt_file_path)
    _assert_all_equal(expected, labels)


def test_cmedia_load_label():
    expected = [
        Label(0.12345, 0.3333, 50),
        Label(1.112, 1.5, 66),
        Label(1.6666, 1.78, 70),
        Label(1.6666, 1.8, 73),
        Label(1.94333, 2.3, 65)
    ]
    gt_file_path = "./tests/resource/gt_files/cmedia_gt_file.csv"
    labels = dset.CMediaStructure.load_label(gt_file_path)
    _assert_all_equal(expected, labels)


def test_medleydb_load_label():
    t_unit = 256 / 44100
    expected = [
        Label(2.461315, 2.46712, 74.312369),
        Label(2.46712, 2.472925, 74.283337),
        Label(2.472925, 2.47873, 74.256145),
        Label(2.47873, 2.484535, 74.235337),
        Label(2.484535, 2.4903, 74.209788)
    ]
    gt_file_path = "./tests/resource/gt_files/medleydb_gt_file.csv"
    labels = dset.MedleyDBStructure.load_label(gt_file_path)
    _assert_all_equal(expected, labels)
