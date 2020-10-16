import pickle

import numpy as np

from omnizart.drum.labels import extract_label_13_inst


def test_extract_label_13_inst():
    test_data = pickle.load(open("tests/resource/drum_test_data.pickle", "rb"))
    m_beat_arr = test_data["m_beat_arr"]
    expected_label = test_data["label"]

    _, label_13 = extract_label_13_inst("tests/resource/drum_test_data.mid", m_beat_arr)
    assert np.array_equal(expected_label, label_13)
