"""Loss functions for Music module."""

# pylint: disable=E1121,E1130

import tensorflow as tf
from tensorflow.python.ops import array_ops


def focal_loss(target_tensor, prediction_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.

    Multi-labels Focal loss formula:

    .. math::

        FL = -\alpha * (z-p)^\gamma * \log{(p)} -(1-\alpha) * p^\gamma * \log{(1-p)}

    Which :math:`\alpha` = 0.25, :math:`\gamma` = 2, p = sigmoid(x), z = target_tensor.

    Parameters
    ----------
    prediction_tensor
        A float tensor of shape [batch_size, num_anchors, num_classes] representing the predicted logits for each
        class.
    target_tensor:
        A float tensor of shape [batch_size, num_anchors, num_classes] representing one-hot encoded classification
        targets.
    weights
        A float tensor of shape [batch_size, num_anchors].
    alpha
        A scalar tensor for focal loss alpha hyper-parameter.
    gamma
        A scalar tensor for focal loss gamma hyper-parameter.

    Returns
    -------
    loss
        A scalar tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p)
    pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = -alpha * (pos_p_sub**gamma) * tf.math.log(  # noqa: E226
        tf.clip_by_value(sigmoid_p, 1e-8, 1.0)
    ) - (1-alpha) * (neg_p_sub**gamma) * tf.math.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))  # noqa: E226

    if weights is not None:
        weights = tf.constant(weights, dtype=per_entry_cross_ent.dtype)
        per_entry_cross_ent *= weights

    return tf.reduce_mean(per_entry_cross_ent)


def q_func(y_true, gamma=0.1, total_chs=22):
    return (1-gamma) * y_true + gamma/total_chs  # noqa: E226


def smooth_loss(y_true, y_pred, gamma=0.15, total_chs=22):
    """Function to compute loss after applying **label-smoothing**."""

    total_chs = min(25, max(total_chs, 5))
    clip_value = lambda v_in: tf.clip_by_value(v_in, 1e-8, 1.0)
    target = clip_value(q_func(y_true, gamma=gamma, total_chs=total_chs))
    neg_target = clip_value(q_func(1 - y_true, gamma=gamma, total_chs=total_chs))
    sigmoid_p = clip_value(tf.nn.sigmoid(y_pred))
    neg_sigmoid_p = clip_value(tf.nn.sigmoid(1 - y_pred))

    cross_entropy = -target * tf.math.log(sigmoid_p) - neg_target * tf.math.log(neg_sigmoid_p)
    return tf.reduce_mean(cross_entropy)
