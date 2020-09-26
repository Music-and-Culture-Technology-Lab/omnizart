import pytest
import numpy as np

from omnizart.music import prediction as putils



def test_cut_frame():
    zeros = np.zeros((20, 10))
    feat = np.ones((20, 352))
    in_feat = np.concatenate([zeros, feat, zeros], axis=1)
    out_feat = putils.cut_frame(in_feat, feature_num=372)
    assert out_feat.shape == (20, 352)
    assert np.array_equiv(out_feat, 1)


def test_cut_batch_pred():
    zeros = np.zeros((25, 30))
    feat = np.ones((50, 30))
    pred = np.concatenate([zeros, feat, zeros])
    b_pred = [pred, pred, pred]
    out_pred = putils.cut_batch_pred(b_pred)
    assert out_pred.shape == (3, 50, 30)
    assert np.array_equiv(out_pred, 1)


@pytest.mark.parametrize("length,b_size", [(100, 3), (1024, 4), (3000, 16), (5000, 8)])
def test_create_batches(length, b_size):
    channels = 4
    feat = np.ones((length, 352, channels))
    timesteps = 128
    expected_len = np.ceil(length/b_size/timesteps)

    out = np.array(putils.create_batches(feat, b_size=b_size, timesteps=timesteps))
    assert out.shape == (expected_len, b_size, timesteps, 384, channels)
