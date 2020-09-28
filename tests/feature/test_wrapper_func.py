import h5py
import numpy as np

from omnizart.feature import wrapper_func as wfunc


def test_get_frame_by_time():
    assert wfunc.get_frame_by_time(150) == 25840
    assert wfunc.get_frame_by_time(77, 16000) == 4812
    assert wfunc.get_frame_by_time(2, 16000, 128) == 250


def test_extract_patch_cqt(mocker):
    with h5py.File("tests/resource/sample_feature.hdf") as fin:
        mini_beat_arr = fin["mini_beat_arr"][:]
        cqt = fin["cqt"][:]
        patch_cqt = fin["patch_cqt"][:]

    mocked_extract_cqt = mocker.patch("omnizart.feature.wrapper_func.extract_cqt")
    mocked_extract_mini_beat = mocker.patch("omnizart.feature.wrapper_func.extract_mini_beat_from_audio_path")
    mocked_extract_cqt.return_value = cqt
    mocked_extract_mini_beat.return_value = mini_beat_arr
    extracted = wfunc.extract_patch_cqt("audio/path", sampling_rate=44100, hop_size=256)

    assert extracted.shape == (1100, 120, 120)
    assert np.array_equiv(patch_cqt, extracted)
