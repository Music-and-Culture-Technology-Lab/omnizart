"""Transcription model of drum leveraging spectral normalization.

The model was originally developed with tensorflow 1.12.
We rewrite the model with tensorflow 2.3 module and uses keras to implement most of
the functionalities for better readability.

Original Author: I-Chieh, Wei
Rewrite by: BreezeWhite
"""
# pylint: disable=C0103,W0221,W0222,W0201,E1120,E1123
import tensorflow as tf

from omnizart.models.utils import shape_list


class SpectralNormalization(tf.keras.layers.Wrapper):
    """Spectral normalization layer.

    Original implementation referes to `here <https://github.com/thisisiron/spectral_normalization-tf2>`_.
    """
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer)
            )
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = shape_list(self.w)

        self.v = self.add_weight(
            shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name='sn_v',
            dtype=tf.float32
        )

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name='sn_u',
            dtype=tf.float32
        )

        super().build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)

        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)


class ConvSN2D(tf.keras.layers.Layer):
    """Just a wrapper layer for using spectral normalization.

    Original implementation referes to `here <https://github.com/thisisiron/spectral_normalization-tf2>`_.
    """
    def __init__(
        self,
        filters,
        kernel_size=(4, 4),
        strides=(2, 2),
        iteration=1,
        eps=1e-12,
        training=True,
        scope="conv_0",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.iteration = iteration
        self.eps = eps
        self.training = training
        self.scope = scope
        with tf.name_scope(scope):
            conv_2d = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, name=f"{scope}_conv_in_sn")
            self.conv_sn_2d = SpectralNormalization(conv_2d, iteration=iteration, eps=eps, training=training, **kwargs)

    def call(self, inputs):
        return self.conv_sn_2d(inputs)

    def get_config(self):
        """This is neccessary to save the model architecture."""
        config = super().get_config().copy()
        config.update(
            {
                "scope": self.scope,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "iteration": self.iteration,
                "eps": self.eps,
                "training": self.training
            }
        )
        return config


def conv_sa(
    x,
    channels,
    kernel=(4, 4),
    strides=(2, 2),
    pad=0,
    pad_type="zero",
    spectral_norm=True,
    scope="conv_0"
):
    with tf.name_scope(scope):
        if pad > 0:
            height = shape_list(x)[1]
            if height % strides[1] == 0:
                pad *= 2
            else:
                pad = max(kernel[1] - (height % strides[1]), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if spectral_norm:
            return ConvSN2D(channels, kernel_size=kernel, strides=strides, scope=f"{scope}_SN")(x)
        return tf.keras.layers.Conv2D(
            channels, kernel_size=kernel, strides=strides, padding='same', name=f"{scope}_conv"
        )(x)


def down_sample(x):
    return tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)


def cnn_attention(x, channels, scope='attention'):
    with tf.name_scope(scope):
        ori_shape = shape_list(x)
        max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")

        out_f = conv_sa(x, channels // 8, kernel=(1, 1), strides=(1, 1), scope="f_attn_conv")
        out_f = max_pooling(out_f)

        out_g = conv_sa(x, channels // 8, kernel=(1, 1), strides=(1, 1), scope='g_attn_conv')

        out_h = conv_sa(x, channels // 2, kernel=(1, 1), strides=(1, 1), scope='h_attn_conv')
        out_h = max_pooling(out_h)

        shape_f = shape_list(out_f)
        shape_g = shape_list(out_g)

        flatten_f = tf.reshape(out_f, shape=[-1, shape_f[1] * shape_f[2], shape_f[3]])
        flatten_g = tf.reshape(out_g, shape=[-1, shape_g[1] * shape_g[2], shape_g[3]])

        attn_out = tf.linalg.matmul(flatten_g, flatten_f, transpose_b=True)
        attn_matrix = tf.keras.activations.softmax(attn_out)

        shape_h = shape_list(out_h)
        flatten_h = tf.reshape(out_h, shape=[-1, shape_h[1] * shape_h[2], shape_h[3]])
        attn_out_2 = tf.linalg.matmul(attn_matrix, flatten_h)

        out = tf.reshape(attn_out_2, shape=[-1, ori_shape[1], ori_shape[2], ori_shape[3] // 2])
        out = conv_sa(out, channels, kernel=(1, 1), strides=(1, 1), scope='out_attn_conv')

        gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        x_att = gamma*out + x  # noqa: E226
        x_gma = gamma * out

        return x_att, x_gma


def residual_block(x, channels, spectral_norm=True, scope="resblock"):
    with tf.name_scope(scope):
        with tf.name_scope("res1"):
            out = conv_sa(
                x,
                channels,
                kernel=(3, 3),
                strides=(1, 1),
                pad=1,
                pad_type='reflect',
                spectral_norm=spectral_norm,
                scope=f"{scope}_res1"
            )
            out = tf.keras.layers.ELU()(out)

        with tf.name_scope("res2"):
            out = conv_sa(
                out,
                channels,
                kernel=(3, 3),
                strides=(1, 1),
                pad=1,
                pad_type="reflect",
                spectral_norm=spectral_norm,
                scope=f"{scope}_res2"
            )
            out = down_sample(out)

        with tf.name_scope("shortcut"):
            out_2 = down_sample(x)
            out_2 = conv_sa(
                out_2, channels, kernel=(1, 1), strides=(1, 1), spectral_norm=spectral_norm, scope=f"{scope}_shortcut"
            )

        return out + out_2


def transpose_residual_block(x, channels, to_down=True, spectral_norm=True, scope='transblock'):
    with tf.name_scope(scope):
        init_channel = shape_list(x)[-1]
        with tf.name_scope('res1'):
            out = tf.keras.layers.ELU()(x)
            out = conv_sa(
                out,
                channels,
                kernel=(3, 3),
                strides=(1, 1),
                pad=1,
                pad_type='reflect',
                spectral_norm=spectral_norm,
                scope=f"{scope}_res1"
            )

        with tf.name_scope('res2'):
            out = tf.keras.layers.ELU()(out)
            out = conv_sa(
                out,
                channels,
                kernel=(3, 3),
                strides=(1, 1),
                pad=1,
                pad_type='reflect',
                spectral_norm=spectral_norm,
                scope=f"{scope}_res2"
            )
            if to_down:
                out = down_sample(out)

        if to_down or init_channel != channels:
            with tf.name_scope('shortcut'):
                x = conv_sa(
                    x, channels, kernel=(1, 1), strides=(1, 1), spectral_norm=spectral_norm, scope=f"{scope}_shortcut"
                )
                if to_down:
                    x = down_sample(x)

        return out + x


def drum_model(out_classes, mini_beat_per_seg, res_block_num=3, channels=64, spectral_norm=True):
    """Get the drum transcription model.

    Constructs the drum transcription model instance for training/inference.

    Parameters
    ----------
    out_classes: int
        Total output classes, refering to classes of drum types.
        Currently there are 13 pre-defined drum percussions.
    mini_beat_per_seg: int
        Number of mini beats in a segment. Can be understood as the range of time
        to be considered for training.
    res_block_num: int
        Number of residual blocks.

    Returns
    -------
    model: tf.keras.Model
        A tensorflow keras model instance.
    """
    with tf.name_scope('transcription_model'):
        inp_wrap = tf.keras.Input(shape=(120, 120, mini_beat_per_seg), name="input_tensor")
        input_tensor = inp_wrap * tf.constant(100)

        padded_input = tf.pad(input_tensor, [[0, 0], [0, 0], [1, 0], [0, 0]], name='tf_diff2_pady')[:, :, :-1]
        pad_out = input_tensor - padded_input
        pad_out = tf.concat([tf.zeros_like(pad_out)[:, :, :1], pad_out[:, :, 1:]], axis=2)
        pad_out = tf.concat([input_tensor, pad_out], axis=-1, name='tf_diff2_concat')

        out = residual_block(pad_out, channels=channels, spectral_norm=spectral_norm, scope='init_resbk')
        out = transpose_residual_block(out, channels=channels, spectral_norm=spectral_norm, scope='fd_resbk')
        x_att, _ = cnn_attention(out, channels=channels, scope='self_attn')

        for i in range(res_block_num):
            if i == 0:
                # first res layer
                out_2 = transpose_residual_block(
                    x_att, channels=channels, spectral_norm=spectral_norm, scope=f"md_resbk_{i}"
                )
            elif i != res_block_num - 1:
                # middle res layer
                out_2 = transpose_residual_block(
                    out_2, channels=channels, spectral_norm=spectral_norm, scope=f"md_resbk_{i}"
                )
            else:
                # last res layer
                out_2 = transpose_residual_block(
                    out_2, channels=channels, spectral_norm=spectral_norm, to_down=False, scope=f'md_resbk_{i}'
                )

        flat_out = tf.reshape(out_2, shape=[-1, tf.math.reduce_prod(shape_list(out_2)[1:])])
        dense_1 = tf.keras.layers.Dense(2**10, activation='elu', name='mdl_nn_mlp_o1')(flat_out)
        dense_2 = tf.keras.layers.Dense(2**10, activation='elu', name='mdl_nn_mlp_o2')(dense_1)
        mix_1 = dense_1 + dense_2*0.25  # noqa: E226

        dense_3 = tf.keras.layers.Dense(2**10, activation='elu', name='mdl_nn_mlp_o3')(mix_1)
        mix_2 = dense_3 + mix_1*0.25  # noqa: E226

        dense_4 = tf.keras.layers.Dense(2**10, activation='elu', name='mdl_nn_mlp_of2')(mix_2)
        dense_5 = tf.keras.layers.Dense(out_classes * mini_beat_per_seg, activation='tanh',
                                        name='mdl_nn_mlp_of3')(dense_4)

        out = dense_5*70 + 50  # noqa: E226
        out = tf.reshape(out, shape=[-1, out_classes, mini_beat_per_seg])

        return tf.keras.Model(inputs=inp_wrap, outputs=out)
