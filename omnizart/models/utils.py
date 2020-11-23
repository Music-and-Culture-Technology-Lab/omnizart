# pylint: disable=C0103

import numpy as np
import tensorflow as tf

from omnizart.utils import midi2freq


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


def matrix_parser(pred):
    """Prediction parser for vocal_frame."""
    output = np.zeros(shape=(pred.shape[0], 2))

    for i, pred_i in enumerate(pred):
        if np.sum(pred_i) != 0:
            output[i][0] = 1
            output[i][1] = midi2freq(np.argmax(pred_i) / 4 + 21)

    return output


def note_res_downsampling(score):
    # filter
    f = [0.1, 0.2, 0.4, 0.2, 0.1]
    r = len(f) // 2

    new_score = np.zeros((score.shape[0], 88))

    pad = np.zeros((new_score.shape[0], 2))
    score = np.concatenate([pad, score], axis=1)

    f_aug = np.tile(f, (new_score.shape[0], 1))

    for i in range(0, 352, 4):
        cent = i + 2
        lower_bound = max(0, cent - r)
        upper_bound = min(353, (cent + 1) + r)
        new_score[:, i // 4] = np.sum(score[:, lower_bound:upper_bound] * f_aug, axis=1)

    return new_score


def padding(x,
            feature_num,
            timesteps,
            dimension=False):

    extended_chorale = np.array(x)

    if ((feature_num - x.shape[1]) % 2) == 0:
        p_t = (feature_num - x.shape[1]) // 2
        p_b = p_t
    else:
        p_t = (feature_num - x.shape[1]) // 2
        p_b = p_t + 1

    top = np.zeros((extended_chorale.shape[0], p_t))
    bottom = np.zeros((extended_chorale.shape[0], p_b))
    extended_chorale = np.concatenate([top, extended_chorale, bottom], axis=1)

    padding_dimensions = (timesteps,) + extended_chorale.shape[1:]

    padding_start = np.zeros(padding_dimensions)
    padding_end = np.zeros(padding_dimensions)

    padding_start[:, :p_t] = 1
    padding_end[:, -p_b:] = 1

    extended_chorale = np.concatenate((padding_start,
                                       extended_chorale,
                                       padding_end),
                                      axis=0)

    if dimension:
        return extended_chorale, p_t, p_b

    return extended_chorale
