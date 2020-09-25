"""Utility functions for Music module"""

import numpy as np
from scipy.special import expit

from omnizart.utils import get_logger


logger = get_logger("Music Prediction")


def cut_frame(frm, ori_feature_size=352, feature_num=384):
    feat_num = frm.shape[1]
    assert feat_num == feature_num

    cut_start = (feat_num-ori_feature_size) // 2  # noqa: E226
    c_range = range(cut_start, cut_start + ori_feature_size)

    return frm[:, c_range]


def cut_batch_pred(b_pred):
    t_len = len(b_pred[0])
    cut_rr = range(round(t_len * 0.25), round(t_len * 0.75))
    cut_pp = []
    for pred in b_pred:
        cut_pp.append(pred[cut_rr])

    return np.array(cut_pp)


def create_batches(feature, b_size, timesteps, feature_num=384):
    frms = np.ceil(len(feature) / timesteps)
    bss = np.ceil(frms / b_size).astype("int")

    pad_bottom = (feature_num - feature.shape[1]) // 2
    pad_top = feature_num - feature.shape[1] - pad_bottom
    f_len = len(feature)
    channel = feature.shape[2]
    pbb = np.zeros((f_len, pad_bottom, channel))
    ptt = np.zeros((f_len, pad_top, channel))
    feature = np.hstack([pbb, feature, ptt])

    batch = []
    for i in range(bss):
        container = np.zeros((b_size, timesteps, feature.shape[1], feature.shape[2]))
        for ii in range(b_size):
            start_i = i*b_size*timesteps + ii*timesteps  # noqa: E226
            if start_i >= f_len:
                break
            end_i = min(start_i + timesteps, len(feature))
            container[ii, 0:(end_i - start_i)] = feature[start_i:end_i]
        batch.append(container)

    return batch


def predict(feature, model, timesteps=128, feature_num=384, batch_size=4):
    """Make predictions on the feature.

    Generate predictions by using the loaded model.

    Parameters
    ----------
    feature: numpy.ndarray
        Extracted feature of the audio.
        Dimension:  timesteps x feature_size x channels
    model: keras.Model
        The loaded model instance
    feature_num: int
        Padding along the feature dimension to the size `feature_num`
    batch_size: int
        Batch size for each step of prediction. The size is depending on the available GPU memory.

    Returns
    -------
    pred: numpy.ndarray
        The predicted results. The values are ranging from 0~1.
    """

    # Create batches of the feature
    features = create_batches(feature, b_size=batch_size, timesteps=timesteps, feature_num=feature_num)

    # Container for the batch prediction
    pred = []

    # Initiate lamda function for latter processing of prediction
    cut_frm = lambda x: cut_frame(x, ori_feature_size=352, feature_num=features[0][0].shape[1])

    t_len = len(features[0][0])
    first_split_start = round(t_len * 0.75)
    second_split_start = t_len + round(t_len * 0.25)

    total_batches = len(features)
    features.insert(0, [np.zeros_like(features[0][0])])
    features.append([np.zeros_like(features[0][0])])
    logger.debug("Total batches: %d", total_batches)
    for i in range(1, total_batches + 1):
        print("batch: {}/{}".format(i, total_batches), end="\r")
        first_half_batch = []
        second_half_batch = []
        b_size = len(features[i])
        features[i] = np.insert(features[i], 0, features[i - 1][-1], axis=0)
        features[i] = np.insert(features[i], len(features[i]), features[i + 1][0], axis=0)
        for ii in range(1, b_size + 1):
            ctx = np.concatenate(features[i][ii - 1:ii + 2], axis=0)

            first_half = ctx[first_split_start:first_split_start + t_len]
            first_half_batch.append(first_half)

            second_half = ctx[second_split_start:second_split_start + t_len]
            second_half_batch.append(second_half)

        p_one = model.predict(np.array(first_half_batch), batch_size=b_size)
        p_two = model.predict(np.array(second_half_batch), batch_size=b_size)
        p_one = cut_batch_pred(p_one)
        p_two = cut_batch_pred(p_two)

        for ii in range(b_size):
            frm = np.concatenate([p_one[ii], p_two[ii]])
            pred.append(cut_frm(frm))

    pred = expit(np.concatenate(pred))  # sigmoid function, mapping the ReLU output value to [0, 1]
    return pred
