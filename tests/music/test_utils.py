import pytest
import numpy as np

from omnizart.music import utils


def test_cut_frame():
    zeros = np.zeros((20, 10))
    feat = np.ones((20, 352))
    in_feat = np.concatenate([zeros, feat, zeros], axis=1)
    out_feat = utils.cut_frame(in_feat, feature_num=372)
    assert out_feat.shape == (20, 352)
    assert np.array_equiv(out_feat, 1)


def test_cut_batch_pred():
    zeros = np.zeros((25, 30))
    feat = np.ones((50, 30))
    pred = np.concatenate([zeros, feat, zeros])
    b_pred = [pred, pred, pred]
    out_pred = utils.cut_batch_pred(b_pred)
    assert out_pred.shape == (3, 50, 30)
    assert np.array_equiv(out_pred, 1)


def generate_pred(frame_size, on_pitch, scale=4, occur_num=3):
    pred = np.zeros((frame_size, 88*scale))
    for idx, pitch in enumerate(on_pitch):
        pitch_range = range(pitch*scale, (pitch+1)*scale)
        occur_pos = np.random.choice(pitch_range, size=occur_num, replace=False)
        pred[idx, occur_pos] = 1
    return pred


def validate_down_sample(out, on_pitch):
    for idx, frm in enumerate(out):
        occur_idx = np.where(frm>0)[0][0]
        assert occur_idx == on_pitch[idx]


def test_roll_down_sample():
    frame_size = 200
    scale = 4
    occur_num = 3
    on_pitch = np.random.randint(88, size=frame_size)

    pred = generate_pred(frame_size, on_pitch, scale=scale, occur_num=occur_num)
    out = utils.roll_down_sample(pred, occur_num=occur_num)

    assert out.shape == (frame_size, 88)
    validate_down_sample(out, on_pitch)

    pred_under_th = generate_pred(frame_size, on_pitch, scale=scale, occur_num=occur_num-1)
    out = utils.roll_down_sample(pred_under_th, occur_num=occur_num)
    assert np.array_equiv(out, 0)


def test_down_sample():
    frame_size = 300
    occur_num = 3
    channels = 10

    preds = []
    on_pitches = []
    for _ in range(channels):
        on_pitch = np.random.randint(88, size=frame_size)
        pred = generate_pred(frame_size, on_pitch, occur_num=occur_num)
        on_pitches.append(on_pitch)
        preds.append(pred)

    preds = np.dstack(preds)
    outs = utils.down_sample(preds, occur_num=occur_num)
    assert outs.shape == (frame_size, 88, channels)
    for idx in range(channels):
        validate_down_sample(outs[:,:,idx], on_pitches[idx])


@pytest.mark.parametrize("length,b_size", [(100, 3), (1024, 4), (3000, 16), (5000, 8)])
def test_create_batches(length, b_size):
    channels = 4
    feat = np.ones((length, 352, channels))
    timesteps = 128
    expected_len = np.ceil(length/b_size/timesteps)

    out = np.array(utils.create_batches(feat, b_size=b_size, timesteps=timesteps))
    assert out.shape == (expected_len, b_size, timesteps, 384, channels)
