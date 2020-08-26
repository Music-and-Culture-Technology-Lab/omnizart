"""Utility functions for Music module"""

import numpy as np


def cut_frame(frm, ori_feature_size=352, feature_num=384):
    feat_num = frm.shape[1]
    assert feat_num == feature_num

    cb = (feat_num - ori_feature_size) // 2
    c_range = range(cb, cb + ori_feature_size)

    return frm[:, c_range]


def cut_batch_pred(b_pred):
    t_len = len(b_pred[0])
    cut_rr = range(round(t_len * 0.25), round(t_len * 0.75))
    cut_pp = []
    for i in range(len(b_pred)):
        cut_pp.append(b_pred[i][cut_rr])

    return np.array(cut_pp)


def create_batches(feature, b_size, timesteps, feature_num=384):
    frms = np.ceil(len(feature) / timesteps)
    bss = np.ceil(frms / b_size).astype("int")

    pb = (feature_num - feature.shape[1]) // 2
    pt = feature_num - feature.shape[1] - pb
    l = len(feature)
    ch = feature.shape[2]
    pbb = np.zeros((l, pb, ch))
    ptt = np.zeros((l, pt, ch))
    feature = np.hstack([pbb, feature, ptt])

    BSS = []
    for i in range(bss):
        bs = np.zeros((b_size, timesteps, feature.shape[1], feature.shape[2]))
        for ii in range(b_size):
            start_i = i * b_size * timesteps + ii * timesteps
            if start_i >= len(feature):
                break
            end_i = min(start_i + timesteps, len(feature))
            length = end_i - start_i

            part = feature[start_i : start_i + length]
            bs[ii, 0:length] = part
        BSS.append(bs)

    return BSS


def roll_down_sample(data, occur_num=3, base=88):
    """Down sample feature size for a single pitch.

    Down sample the feature size from 354 to 88 for infering the notes.

    Parameters
    ----------
    data : 2D numpy array
        The thresholded 2D prediction..
    occur_num : int
        For each pitch, the original prediction expands 4 bins wide. This value determines how many positive bins
        should there be to say there is a real activation after down sampling.
    base : 88
        Should be constant as there are 88 pitches on the piano.
    
    Returns
    -------
    return_v : 2D numpy array
        Down sampled prediction.

    Warnings
    --------
    The parameter `data` should be thresholded!
    """

    total_roll = data.shape[1]
    assert total_roll % base == 0, "Wrong length: {}, {} % {} should be zero!".format(total_roll, total_roll, base)

    scale = round(total_roll / base)
    assert occur_num > 0 and occur_num < scale

    return_v = np.zeros((len(data), base), dtype=int)

    for i in range(0, data.shape[1], scale):
        total = np.sum(data[:, i : i + scale], axis=1)
        return_v[:, int(i / scale)] = np.where(total >= occur_num, total / occur_num, 0)
    return_v = np.where(return_v > 1, 1, return_v)

    return return_v


def down_sample(pred, occur_num=3):
    """Down sample multi-channel predictions along the feature dimension.

    Down sample the feature size from 354 to 88 for infering the notes from a multi-channel prediction.
    """
    dd = roll_down_sample(pred[:, :, 0], occur_num=occur_num)
    for i in range(1, pred.shape[2]):
        dd = np.dstack([dd, roll_down_sample(pred[:, :, i], occur_num=occur_num)])

    return dd
