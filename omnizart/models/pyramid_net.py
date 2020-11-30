# pylint: disable=E1120,R0901
import time
import math

import tensorflow as tf


class ShakeDrop(tf.keras.layers.Layer):
    """Shake drop layer.

    Most of the code follows the implementation from tensorflow research [1]_.

    References
    ----------
    .. [1] https://github.com/tensorflow/models/blob/master/research/autoaugment/shake_drop.py
    """
    # pylint: disable=E1123
    def __init__(self, prob, min_alpha=-1, max_alpha=1, min_beta=0, max_beta=1, **kwargs):
        super().__init__(**kwargs)

        self.prob = prob
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_beta = min_beta
        self.max_beta = max_beta

    def call(self, inputs, is_training=True):  # pylint: disable=W0221
        if is_training:
            in_shape = tf.shape(inputs)
            random_tensor = self.prob
            random_tensor += tf.random.uniform(in_shape, dtype=tf.float32)
            binary_tensor = tf.floor(random_tensor)

            alpha_values = tf.random.uniform([in_shape[0], 1, 1, 1], minval=self.min_alpha, maxval=self.max_alpha)
            beta_values = tf.random.uniform([in_shape[0], 1, 1, 1], minval=self.min_beta, maxval=self.max_beta)
            rand_forward = binary_tensor + alpha_values - binary_tensor * alpha_values
            rand_backward = binary_tensor + beta_values - binary_tensor * beta_values
            outputs = inputs * rand_backward + tf.stop_gradient(
                inputs*rand_forward - inputs*rand_backward  # noqa: E226
            )
            return outputs

        expected_alpha = (self.min_alpha + self.max_alpha) / 2
        return (self.prob + expected_alpha - self.prob * expected_alpha) * inputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "prob": self.prob,
            "min_alpha": self.min_alpha,
            "max_alpha": self.max_alpha,
            "min_beta": self.min_beta,
            "max_beta": self.max_beta
        })
        return config


class PyramidBlock(tf.keras.layers.Layer):
    """Pyramid block for building pyramid net."""
    def __init__(self, out_channel, stride=1, padding="same", prob=1.0, shakedrop=True, **kwargs):
        super().__init__(**kwargs)

        self.shakedrop = shakedrop
        self.padding = padding
        self.stride = stride
        self.prob = prob
        self.downsample = stride == 2
        self.out_channel = int(out_channel)

        conv_init = tf.keras.initializers.VarianceScaling(scale=0.1, seed=int(time.time()))

        self.avgpool = tf.keras.layers.AveragePooling2D(strides=2, padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.conv_1 = tf.keras.layers.Conv2D(
            out_channel, 3, strides=stride, padding=padding, kernel_initializer=conv_init, activation='relu'
        )
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv_2 = tf.keras.layers.Conv2D(
            out_channel, 3, strides=1, padding=padding, use_bias=False, kernel_initializer=conv_init, activation='relu'
        )
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()

        if shakedrop:
            self.shakedrop_layer = ShakeDrop(prob)

    def call(self, inputs, is_training=True):  # pylint: disable=W0221
        res = self._shortcut(inputs)
        output = self.batch_norm_1(inputs)
        output = self.conv_1(output)
        output = self.batch_norm_2(output)
        output = self.relu(output)
        output = self.conv_2(output)
        output = self.batch_norm_3(output)
        if self.shakedrop:
            output = self.shakedrop_layer(output, is_training=is_training)

        return output + res

    def _shortcut(self, inputs):
        out = inputs
        if self.downsample:
            out = self.avgpool(inputs)

        num_filters = inputs.shape[3]
        if num_filters != self.out_channel:
            diff = self.out_channel - num_filters
            assert diff > 0
            padding = [[0, 0], [0, 0], [0, 0], [0, diff]]
            out = tf.pad(out, padding)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "prob": self.prob,
            "out_channel": self.out_channel,
            "stride": self.stride,
            "padding": self.padding,
            "shakedrop": self.shakedrop
        })
        return config


def _make_blocks(num_blocks, kernel_sizes, probs, stride=2, shakedrop=True):
    assert len(kernel_sizes) == num_blocks, f"{num_blocks} {len(kernel_sizes)}"
    assert len(probs) == num_blocks, f"{num_blocks} {len(probs)}"
    blocks = []
    for kernel_size, prob in zip(kernel_sizes, probs):
        blocks.append(PyramidBlock(kernel_size, prob=prob, shakedrop=shakedrop, stride=stride))
        stride = 1
    return blocks


class PyramidNet(tf.keras.Model):
    """Pyramid Net with shake drop layer."""
    def __init__(
        self,
        out_classes=6,
        min_kernel_size=16,
        depth=110,
        alpha=270,
        shakedrop=True,
        semi_loss_weight=1,
        semi_xi=1e-6,
        semi_epsilon=40,
        semi_iters=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.out_classes = out_classes
        self.min_kernel_size = min_kernel_size
        self.depth = depth
        self.alpha = alpha
        self.shakedrop = shakedrop
        self.semi_loss_weight = semi_loss_weight
        self.semi_xi = semi_xi
        self.semi_epsilon = semi_epsilon
        self.semi_iters = semi_iters

        self.kl_loss = tf.keras.losses.KLDivergence()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        if (depth - 2) % 6 != 0:
            raise ValueError(f"Value of 'depth' should be one of [20, 32, 44, 56, 110, 1202]. Received: {depth}.")

        n_units = (depth - 2) // 6
        self.kernel_sizes = [min_kernel_size] + list(map(
            lambda x: math.ceil(alpha * (x + 1)) / (3 * n_units) + min_kernel_size,
            list(range(n_units * 3))
        ))

        self.conv_1 = tf.keras.layers.Conv2D(
            self.kernel_sizes[0],
            (7, 7),
            strides=(2, 2),
            use_bias=False,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal()
        )
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(name="batch_norm_1")
        self.relu_1 = tf.keras.layers.ReLU(name="relu_1")
        self.maxpool = tf.keras.layers.MaxPool2D(strides=2, name="max_pool")
        self.batch_norm_out = tf.keras.layers.BatchNormalization(name="batch_norm_2")

        total_blocks = n_units * 3
        calc_prob = lambda cur_layer: 1 - (cur_layer + 1) / total_blocks * 0.5
        self.kernel_sizes = self.kernel_sizes[1:]
        self.blocks = _make_blocks(
            n_units,
            kernel_sizes=self.kernel_sizes[:n_units],
            probs=[calc_prob(idx) for idx in range(n_units)],
            shakedrop=shakedrop,
            stride=1
        )
        self.blocks += _make_blocks(
            n_units,
            kernel_sizes=self.kernel_sizes[n_units:n_units * 2],
            probs=[calc_prob(idx) for idx in range(n_units, n_units * 2)],
            shakedrop=shakedrop,
            stride=2
        )
        self.blocks += _make_blocks(
            n_units,
            kernel_sizes=self.kernel_sizes[n_units*2:n_units*3],  # noqa: E226
            probs=[calc_prob(idx) for idx in range(n_units*2, n_units*3)],  # noqa: E226
            shakedrop=shakedrop,
            stride=2
        )

        self.relu_out = tf.keras.layers.ReLU(name="relu_out")
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(1, 11), name="avg_pool")
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.dense = tf.keras.layers.Dense(out_classes * 19, activation='sigmoid', name="dense_out")
        self.reshape = tf.keras.layers.Reshape((19, out_classes))

    def call(self, inputs, is_training=True):  # pylint: disable=W0221
        enc = self.conv_1(inputs)
        enc = self.batch_norm_1(enc)
        enc = self.relu_1(enc)
        b_out = self.maxpool(enc)

        for block in self.blocks:
            b_out = block(b_out, is_training=is_training)

        output = self.relu_out(b_out)
        output = self.avgpool(output)
        output = self.flatten(output)
        output = self.dense(output)
        return self.reshape(output)

    def train_step(self, data):
        data1, data2 = data
        semi = False
        if isinstance(data1, tuple):
            # Semi-supervise learning
            assert isinstance(data2, tuple)
            super_feat, super_label = data1
            unsup_feat, _ = data2
            semi = True
        else:
            super_feat = data1
            super_label = data2

        with tf.GradientTape() as tape:
            super_pred = self(super_feat, is_training=True)
            loss = self._compute_supervised_loss(super_label, super_pred)
            if semi:
                loss += self._compute_unsupervised_loss(unsup_feat) * self.semi_loss_weight

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        grads = self.optimizer._clip_gradients(grads)  # pylint: disable=protected-access
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(super_label, super_pred)
        result = {m.name: m.result() for m in self.metrics}
        result.update({"loss": self.loss_tracker.result()})
        return result

    def test_step(self, data):
        data1, data2 = data
        semi = False
        if isinstance(data1, tuple):
            # Semi-supervise learning
            assert isinstance(data2, tuple)
            super_feat, super_label = data1
            unsup_feat, _ = data2
            semi = True
        else:
            super_feat = data1
            super_label = data2

        super_pred = self(super_feat, is_training=False)
        loss = self._compute_supervised_loss(super_label, super_pred)
        if semi:
            loss += self._compute_unsupervised_loss(unsup_feat) * self.semi_loss_weight

        self.compiled_metrics.update_state(super_label, super_pred)
        self.loss_tracker.update_state(loss)
        result = {m.name: m.result() for m in self.metrics}
        result.update({"loss": self.loss_tracker.result()})
        return result

    def _compute_supervised_loss(self, label, pred):
        loss = self.compiled_loss(label, pred)
        empahsize_channel = [1, 2, 4]
        weight = 0.7
        emp_loss = 0
        for channel in empahsize_channel:
            emp_loss += self.compiled_loss(label[:, :, channel], pred[:, :, channel])
        return loss * (1 - weight) + emp_loss * weight

    def _compute_unsupervised_loss(self, unsup_feat):
        """Computes VAT loss.

        Original implementations are from [1]_.

        References
        ----------
        .. [1] https://github.com/takerum/vat_tf
        """
        unsup_pred = self(unsup_feat)
        r_adv = self._gen_virtual_adv_perturbation(unsup_feat, unsup_pred)
        tf.stop_gradient(unsup_pred)
        unsup_pred_copy = unsup_pred
        adv_pred = self(unsup_feat + r_adv)
        loss = self.kl_loss(unsup_pred_copy, adv_pred)
        return tf.identity(loss)

    def _gen_virtual_adv_perturbation(self, unsup_feat, unsup_pred):
        self._switch_batch_norm_trainable_stat()

        perturb = tf.random.normal(tf.shape(unsup_feat))
        for _ in range(self.semi_iters):
            perturb = self.semi_xi * _normalize(perturb)
            unsup_pred_copy = unsup_pred
            perturb_pred = self(unsup_feat + perturb)
            dist = self.kl_loss(unsup_pred_copy, perturb_pred)
            grad = tf.gradients(dist, [perturb], aggregation_method=2)[0]
            perturb_pred = tf.stop_gradient(grad)

        self._switch_batch_norm_trainable_stat()
        return self.semi_epsilon * _normalize(perturb)

    def _switch_batch_norm_trainable_stat(self):
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable ^= True

    def get_config(self):
        return {
            "class_name": self.__class__.__name__,
            "config": {
                "out_classes": self.out_classes,
                "min_kernel_size": self.min_kernel_size,
                "depth": self.depth,
                "alpha": self.alpha,
                "shakedrop": self.shakedrop,
                "semi_loss_weight": self.semi_loss_weight,
                "semi_xi": self.semi_xi,
                "semi_epsilon": self.semi_epsilon,
                "semi_iters": self.semi_iters
            }
        }


def _normalize(tensor):
    tensor /= (1e-12 + tf.reduce_max(tf.abs(tensor), range(1, len(tensor.shape)), keepdims=True))
    tensor /= tf.sqrt(1e-6 + tf.reduce_sum(tensor**2, range(1, len(tensor.shape)), keepdims=True))
    return tensor
