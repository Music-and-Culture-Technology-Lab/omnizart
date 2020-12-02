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


def note_res_downsampling(score):
    note_filter = [0.1, 0.2, 0.4, 0.2, 0.1]
    cent_res = len(note_filter) // 2

    new_score = np.zeros((len(score), 88))

    pad = np.zeros((len(new_score), 2))
    score = np.concatenate([pad, score], axis=1)

    note_filter_aug = np.tile(note_filter, (len(new_score), 1))

    for i in range(0, 352, 4):
        cent = i + 2
        lower_bound = max(0, cent - cent_res)
        upper_bound = min(353, (cent + 1) + cent_res)
        new_score[:, i // 4] = np.sum(score[:, lower_bound:upper_bound] * note_filter_aug, axis=1)

    return new_score


def padding(x, feature_num, timesteps, dimension=False):

    extended_chorale = np.array(x)

    if ((feature_num - x.shape[1]) % 2) == 0:
        p_t = (feature_num - x.shape[1]) // 2
        p_b = p_t
    else:
        p_t = (feature_num - x.shape[1]) // 2
        p_b = p_t + 1

    top = np.zeros((len(extended_chorale), p_t))
    bottom = np.zeros((len(extended_chorale), p_b))
    extended_chorale = np.concatenate([top, extended_chorale, bottom], axis=1)

    padding_dimensions = (timesteps,) + extended_chorale.shape[1:]

    padding_start = np.zeros(padding_dimensions)
    padding_end = np.zeros(padding_dimensions)

    padding_start[:, :p_t] = 1
    padding_end[:, -p_b:] = 1

    extended_chorale = np.concatenate(
        (padding_start, extended_chorale, padding_end), axis=0
    )

    if dimension:
        return extended_chorale, p_t, p_b

    return extended_chorale
