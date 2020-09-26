import numpy as np

from omnizart.music import inference as inf


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
    out = inf.roll_down_sample(pred, occur_num=occur_num)

    assert out.shape == (frame_size, 88)
    validate_down_sample(out, on_pitch)

    pred_under_th = generate_pred(frame_size, on_pitch, scale=scale, occur_num=occur_num-1)
    out = inf.roll_down_sample(pred_under_th, occur_num=occur_num)
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
    outs = inf.down_sample(preds, occur_num=occur_num)
    assert outs.shape == (frame_size, 88, channels)
    for idx in range(channels):
        validate_down_sample(outs[:,:,idx], on_pitches[idx])


def test_find_occur():
    data = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])
    expected = [{"onset": 0, "offset": 1}, {"onset": 7, "offset": 11}, {"onset": 13, "offset": 15}]
    out = inf.find_occur(data)
    assert out == expected

    out = inf.find_occur(data, min_duration=0.01)
    assert out == expected

    out = inf.find_occur(data, t_unit=0.01)
    assert out == [{"onset": 7, "offset": 11}, {"onset": 13, "offset": 15}]

    out = inf.find_occur(data, min_duration=0.07)
    assert out == [{"onset": 7, "offset": 11}]
    assert inf.find_occur(np.array([0, 0, 0])) == []


def test_find_min_max_stren():
    stren = np.random.random(500)
    max_v, min_v = np.max(stren), np.min(stren)
    notes = [{"stren": stv} for stv in stren]
    assert inf.find_min_max_stren(notes) == (min_v, max_v)
    assert inf.find_min_max_stren([]) == (0.5, 0.5)


def test_infer_pitch():
    zeros = np.zeros(50)
    onset = np.array([
        0, 2.5, 3.1, 3, 3, 2.8, 2, 3.3, 3, 2.9, 2.8, 2.8, 1, 0,  # Two peaks, but filter-out the first one due to distance too close
        0, 0, 2, 2.6, 3, 3, 3.1, 3, 1.2, 1.4, 0, 0, 0, 0,  # One peak
        0.3, 0.2, 0, 0.1, 0, 0, 2.3, 2.4, 2.5, 2.6, 2.2, 2,  # One peak, will be eliminated due to duration too short
        0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0
    ])
    dura = np.array([
        0, 0, 0.5, 0.8, 1.8, 2, 2, 0.6, 1.5, 1.3, 1.4, 1.9, 2.5, 2.6,
        0, 0, 0, 0, 0, 0.2, 0.3, 0.8, 1.2, 1.4, 1.8, 2, 2, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.8, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0
    ])
    expected = [
        {"start": 5, "end": 18, "stren": 3.3},
        {"start": 18, "end": 26, "stren": 3.1}
    ]

    pitch = np.stack([zeros, dura, onset], axis=1)
    out = inf.infer_pitch(pitch)
    assert out == expected
    assert inf.infer_pitch(pitch, shortest=30) == []

    pitch[:, 2] = 0
    assert inf.infer_pitch(pitch) == []
