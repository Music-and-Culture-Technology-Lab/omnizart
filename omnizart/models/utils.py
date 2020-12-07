# pylint: disable=C0103

import numpy as np
import tensorflow as tf

from librosa.core import midi_to_hz


def shape_list(input_tensor):
    """Return list of dims, statically where possible."""
    tensor = tf.convert_to_tensor(input_tensor)

    # If unknown rank, return dynamic shape
    if tensor.get_shape().dims is None:
        return tf.shape(tensor)

    static = tensor.get_shape().as_list()
    shape = tf.shape(tensor)

    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def get_contour(pred):
    output = np.zeros(shape=(pred.shape[0], 2))

    for i, pred_i in enumerate(pred):
        if np.sum(pred_i) != 0:
            output[i][0] = 1
            output[i][1] = midi_to_hz(np.argmax(pred_i) / 4 + 21)

    return output


def padding(feature, feature_num, timesteps, dimension=False):

    f_len = len(feature)

    if ((feature_num - feature.shape[1]) % 2) == 0:
        pad_top = (feature_num - feature.shape[1]) // 2
        pad_bottom = pad_top
    else:
        pad_top = (feature_num - feature.shape[1]) // 2
        pad_bottom = pad_top + 1

    pbb = np.zeros((f_len, pad_bottom))
    ptt = np.zeros((f_len, pad_top))
    feature = np.hstack([ptt, feature, pbb])

    padding_dimensions = (timesteps,) + feature.shape[1:]

    padding_start = np.zeros(padding_dimensions)
    padding_end = np.zeros(padding_dimensions)

    padding_start[:, :pad_top] = 1
    padding_end[:, -pad_bottom:] = 1

    feature = np.vstack([padding_start, feature, padding_end])

    if dimension:
        return feature, pad_top, pad_bottom

    return feature
