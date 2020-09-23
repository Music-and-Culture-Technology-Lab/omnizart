"""Definition of customized U-Net like model architecture."""

# pylint: disable=C0103,R0914,R0915,W0221

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Activation,
    Dropout,
    Conv2D,
    Conv2DTranspose,
    Add,
    Concatenate
)
from omnizart.models.t2t import local_attention_2d, split_heads_2d, combine_heads_2d


def conv_block(input_tensor, channel, kernel_size, strides=(2, 2), dilation_rate=1, dropout_rate=0.4):
    """Convolutional encoder block of U-net.

    The block is a fully convolutional block. The encoder block does not downsample the input feature,
    and thus the output will have the same dimension as the input.
    """

    skip = input_tensor

    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2D(
        channel, kernel_size, strides=strides, dilation_rate=dilation_rate, padding="same"
    )(input_tensor)

    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2D(
        channel, kernel_size, strides=(1, 1), dilation_rate=dilation_rate, padding="same"
    )(input_tensor)

    if strides != (1, 1):
        skip = Conv2D(channel, (1, 1), strides=strides, padding="same")(skip)
    input_tensor = Add()([input_tensor, skip])

    return input_tensor


def transpose_conv_block(input_tensor, channel, kernel_size, strides=(2, 2), dropout_rate=0.4):
    skip = input_tensor

    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2D(channel, kernel_size, strides=(1, 1), padding="same")(input_tensor)

    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2DTranspose(channel, kernel_size, strides=strides, padding="same")(input_tensor)

    if strides != (1, 1):
        skip = Conv2DTranspose(channel, (1, 1), strides=strides, padding="same")(skip)
    input_tensor = Add()([input_tensor, skip])

    return input_tensor


def semantic_segmentation(
    feature_num=352, timesteps=256, multi_grid_layer_n=1, multi_grid_n=3, ch_num=1, out_class=2
):
    """Improved U-net model with Atrous Spatial Pyramid Pooling (ASPP) block."""
    layer_out = []

    input_score = Input(shape=(timesteps, feature_num, ch_num), name="input_score_48")
    en = Conv2D(2**5, (7, 7), strides=(1, 1), padding="same")(input_score)
    layer_out.append(en)

    en_l1 = conv_block(en, 2**5, (3, 3), strides=(2, 2))
    en_l1 = conv_block(en_l1, 2**5, (3, 3), strides=(1, 1))
    layer_out.append(en_l1)

    en_l2 = conv_block(en_l1, 2**6, (3, 3), strides=(2, 2))
    en_l2 = conv_block(en_l2, 2**6, (3, 3), strides=(1, 1))
    en_l2 = conv_block(en_l2, 2**6, (3, 3), strides=(1, 1))
    layer_out.append(en_l2)

    en_l3 = conv_block(en_l2, 2**7, (3, 3), strides=(2, 2))
    en_l3 = conv_block(en_l3, 2**7, (3, 3), strides=(1, 1))
    en_l3 = conv_block(en_l3, 2**7, (3, 3), strides=(1, 1))
    en_l3 = conv_block(en_l3, 2**7, (3, 3), strides=(1, 1))
    layer_out.append(en_l3)

    en_l4 = conv_block(en_l3, 2**8, (3, 3), strides=(2, 2))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    layer_out.append(en_l4)

    feature = en_l4

    for _ in range(multi_grid_layer_n):
        feature = BatchNormalization()(Activation("relu")(feature))
        feature = Dropout(0.3)(feature)
        m = BatchNormalization()(Conv2D(2**9, (1, 1), strides=(1, 1), padding="same", activation="relu")(feature))
        multi_grid = m
        for ii in range(multi_grid_n):
            m = BatchNormalization()(
                Conv2D(2**9, (3, 3), strides=(1, 1), dilation_rate=2**ii, padding="same", activation="relu")(feature)
            )
            multi_grid = Concatenate()([multi_grid, m])
        multi_grid = Dropout(0.3)(multi_grid)
        feature = Conv2D(2**9, (1, 1), strides=(1, 1), padding="same")(multi_grid)
        layer_out.append(feature)

    feature = BatchNormalization()(Activation("relu")(feature))

    feature = Conv2D(2**8, (1, 1), strides=(1, 1), padding="same")(feature)
    feature = Add()([feature, en_l4])
    de_l1 = transpose_conv_block(feature, 2**7, (3, 3), strides=(2, 2))
    layer_out.append(de_l1)

    skip = de_l1
    de_l1 = BatchNormalization()(Activation("relu")(de_l1))
    de_l1 = Concatenate()([de_l1, BatchNormalization()(Activation("relu")(en_l3))])
    de_l1 = Dropout(0.4)(de_l1)
    de_l1 = Conv2D(2**7, (1, 1), strides=(1, 1), padding="same")(de_l1)
    de_l1 = Add()([de_l1, skip])
    de_l2 = transpose_conv_block(de_l1, 2**6, (3, 3), strides=(2, 2))
    layer_out.append(de_l2)

    skip = de_l2
    de_l2 = BatchNormalization()(Activation("relu")(de_l2))
    de_l2 = Concatenate()([de_l2, BatchNormalization()(Activation("relu")(en_l2))])
    de_l2 = Dropout(0.4)(de_l2)
    de_l2 = Conv2D(2**6, (1, 1), strides=(1, 1), padding="same")(de_l2)
    de_l2 = Add()([de_l2, skip])
    de_l3 = transpose_conv_block(de_l2, 2**5, (3, 3), strides=(2, 2))
    layer_out.append(de_l3)

    skip = de_l3
    de_l3 = BatchNormalization()(Activation("relu")(de_l3))
    de_l3 = Concatenate()([de_l3, BatchNormalization()(Activation("relu")(en_l1))])
    de_l3 = Dropout(0.4)(de_l3)
    de_l3 = Conv2D(2**5, (1, 1), strides=(1, 1), padding="same")(de_l3)
    de_l3 = Add()([de_l3, skip])
    de_l4 = transpose_conv_block(de_l3, 2**5, (3, 3), strides=(2, 2))
    layer_out.append(de_l4)

    de_l4 = BatchNormalization()(Activation("relu")(de_l4))
    de_l4 = Dropout(0.4)(de_l4)
    out = Conv2D(out_class, (1, 1), strides=(1, 1), padding="same", name="prediction")(de_l4)

    return Model(inputs=input_score, outputs=out)


class MultiHeadAttention(tf.keras.layers.Layer):
    """Attention layer for 2D input feature.

    As the attention mechanism consumes a large amount of memory, here we leverage a divide-and-conquer approach
    implemented in the ``tensor2tensor`` repository. The input feature is first partitioned into smaller parts before
    being passed to do self-attention computation. The processed outputs are then assembled back into the same
    size as the input.

    Parameters
    ----------
    out_channel: int
        Number of output channels.
    d_model: int
        Dimension of embeddings for each position of input feature.
    n_heads: int
        Number of heads for multi-head attention computation. Should be division of `d_model`.
    query_shape: Tuple(int, int)
        Size of each partition.
    memory_flange: Tuple(int, int)
        Additional overlapping size to be extended to each partition, indicating the final size to be
        computed is: (query_shape[0]+memory_flange[0]) x (query_shape[1]+memory_flange[1])

    References
    ----------
    This approach is originated from [1]_.

    .. [1] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran,
       “Image Transformer,” in Proceedings of the 35th International Conference on Machine Learning (ICML), 2018
    """
    def __init__(self, out_channel=64, d_model=32, n_heads=8, query_shape=(128, 24), memory_flange=(8, 8), **kwargs):
        super().__init__(**kwargs)

        self.out_channel = out_channel
        self.d_model = d_model
        self.n_heads = n_heads
        self.query_shape = query_shape
        self.memory_flange = memory_flange

        self.q_conv = Conv2D(d_model, (3, 3), strides=(1, 1), padding="same", name="gen_q_conv")
        self.k_conv = Conv2D(d_model, (3, 3), strides=(1, 1), padding="same", name="gen_k_conv")
        self.v_conv = Conv2D(d_model, (3, 3), strides=(1, 1), padding="same", name="gen_v_conv")
        self.out_conv = Conv2D(out_channel, (3, 3), strides=(1, 1), padding="same", use_bias=False)

    def call(self, inputs):
        q = self.q_conv(inputs)
        k = self.k_conv(inputs)
        v = self.v_conv(inputs)

        q = split_heads_2d(q, self.n_heads)
        k = split_heads_2d(k, self.n_heads)
        v = split_heads_2d(v, self.n_heads)

        k_depth_per_head = self.d_model // self.n_heads
        q *= k_depth_per_head**-0.5

        output = local_attention_2d(q, k, v, query_shape=self.query_shape, memory_flange=self.memory_flange)
        output = combine_heads_2d(output)
        return self.out_conv(output)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "out_channel": self.out_channel,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "query_shape": self.query_shape,
                "memory_flange": self.memory_flange,
            }
        )
        return config


def semantic_segmentation_attn(feature_num=352, timesteps=256, ch_num=1, out_class=2):
    """Customized attention U-net model."""
    layer_out = []

    input_score = Input(shape=(timesteps, feature_num, ch_num), name="input_score_48")
    en = Conv2D(2**5, (7, 7), strides=(1, 1), padding="same")(input_score)
    layer_out.append(en)

    en_l1 = conv_block(en, 2**5, (3, 3), strides=(2, 2))
    en_l1 = conv_block(en_l1, 2**5, (3, 3), strides=(1, 1))
    layer_out.append(en_l1)

    en_l2 = conv_block(en_l1, 2**6, (3, 3), strides=(2, 2))
    en_l2 = conv_block(en_l2, 2**6, (3, 3), strides=(1, 1))
    en_l2 = conv_block(en_l2, 2**6, (3, 3), strides=(1, 1))
    layer_out.append(en_l2)

    en_l3 = conv_block(en_l2, 2**7, (3, 3), strides=(2, 2))
    en_l3 = conv_block(en_l3, 2**7, (3, 3), strides=(1, 1))
    en_l3 = conv_block(en_l3, 2**7, (3, 3), strides=(1, 1))
    en_l3 = conv_block(en_l3, 2**7, (3, 3), strides=(1, 1))
    layer_out.append(en_l3)

    en_l4 = conv_block(en_l3, 2**8, (3, 3), strides=(2, 2))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    layer_out.append(en_l4)

    feature = en_l4
    f_attn = MultiHeadAttention(name="attn_1", out_channel=64, query_shape=(100, 32))(feature)
    f_attn = BatchNormalization()(Activation("relu")(f_attn))
    f_attn = MultiHeadAttention(name="attn-2", d_model=64, out_channel=128, query_shape=(64, 16))(f_attn)
    f_attn = BatchNormalization()(Activation("relu")(f_attn))
    feature = Concatenate()([feature, f_attn])

    feature = Conv2D(2**8, (1, 1), strides=(1, 1), padding="same")(feature)
    feature = Add()([feature, en_l4])
    de_l1 = transpose_conv_block(feature, 2**7, (3, 3), strides=(2, 2))
    layer_out.append(de_l1)

    skip = de_l1
    de_l1 = BatchNormalization()(Activation("relu")(de_l1))
    de_l1 = Concatenate()([de_l1, BatchNormalization()(Activation("relu")(en_l3))])
    de_l1 = Dropout(0.4)(de_l1)
    de_l1 = Conv2D(2**7, (1, 1), strides=(1, 1), padding="same")(de_l1)
    de_l1 = Add()([de_l1, skip])
    de_l2 = transpose_conv_block(de_l1, 2**6, (3, 3), strides=(2, 2))
    layer_out.append(de_l2)

    skip = de_l2
    de_l2 = BatchNormalization()(Activation("relu")(de_l2))
    # en_l2 = MultiHead_Attention(en_l2, query_shape=(25, 16))
    de_l2 = Concatenate()([de_l2, BatchNormalization()(Activation("relu")(en_l2))])
    de_l2 = Dropout(0.4)(de_l2)
    de_l2 = Conv2D(2**6, (1, 1), strides=(1, 1), padding="same")(de_l2)
    de_l2 = Add()([de_l2, skip])
    de_l3 = transpose_conv_block(de_l2, 2**6, (3, 3), strides=(2, 2))
    layer_out.append(de_l3)

    skip = de_l3
    de_l3 = BatchNormalization()(Activation("relu")(de_l3))
    # en_l1 = MultiHead_Attention(en_l1, query_shape=(25, 16))
    de_l3 = Concatenate()([de_l3, BatchNormalization()(Activation("relu")(en_l1))])
    de_l3 = Dropout(0.4)(de_l3)
    de_l3 = Conv2D(2**6, (1, 1), strides=(1, 1), padding="same")(de_l3)
    de_l3 = Add()([de_l3, skip])
    de_l4 = transpose_conv_block(de_l3, 2**6, (3, 3), strides=(2, 2))
    layer_out.append(de_l4)

    de_l4 = BatchNormalization()(Activation("relu")(de_l4))
    de_l4 = Dropout(0.4)(de_l4)
    out = Conv2D(out_class, (1, 1), strides=(1, 1), padding="same", name="prediction")(de_l4)

    return Model(inputs=input_score, outputs=out)
