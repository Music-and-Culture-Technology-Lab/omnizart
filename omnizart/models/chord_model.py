# pylint: disable=W0102,W0221
import tensorflow as tf
from tensorflow.python.framework import ops

from omnizart.models.t2t import positional_encoding, MultiHeadAttention
from omnizart.models.utils import shape_list


class FeedForward(tf.keras.layers.Layer):
    """Feedfoward layer of the transformer model.

    Paramters
    ---------
    n_units: list[int, int]
        A two-element integer list. The first integer represents the output embedding size
        of the first convolution layer, and the second integer represents the embedding size
        of the second convolution layer.
    activation_func: str
        Activation function of the first covolution layer. Available options can be found
        from the tensorflow.keras official site.
    dropout_rate: float
        Dropout rate of all dropout layers.
    """
    def __init__(self, n_units=[2048, 512], activation_func="relu", dropout_rate=0):
        super().__init__()

        self.n_units = n_units
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate

        self.conv_1 = tf.keras.layers.Conv1D(n_units[0], kernel_size=1, activation=activation_func)
        self.conv_2 = tf.keras.layers.Conv1D(n_units[1], kernel_size=1)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inp):
        outputs = self.conv_1(inp)
        outputs = self.conv_2(outputs)
        outputs = self.dropout(outputs)
        outputs += inp
        return self.layer_norm(outputs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "activation_func": self.activation_func,
                "n_units": self.n_units,
                "dropout_rate": self.dropout_rate
            }
        )
        return config


class EncodeSegmentTime(tf.keras.layers.Layer):
    """Encode feature along the time axis.

    Parameters
    ----------
    n_units: int
        Output embedding size.
    n_steps: int
        Time length of the feature.
    segment_width: int
        Context width of each frame. Nearby frames will be concatenated to the feature axis.
        Default to 21, which means past 10 frames and future 10 frames will be concatenated
        to the current frame, resulting a feature dimenstion of *segment_width x freq_size*.
    freq_size: int
        Feature size of the input representation.
    dropout_rate: float
        Dropout rate of all dropout layers.
    """
    def __init__(self, n_units=512, dropout_rate=0, n_steps=100, freq_size=24, segment_width=21):
        super().__init__()

        self.n_steps = n_steps
        self.freq_size = freq_size
        self.segment_width = segment_width
        self.n_units = n_units
        self.dropout_rate = dropout_rate

        self.attn_layer = MultiHeadAttention(
            n_units=freq_size,
            n_heads=2,
            activation_func="relu",
            relative_position=True,
            max_dist=4,
            dropout_rate=dropout_rate
        )
        self.feed_forward = FeedForward(n_units=[freq_size * 4, freq_size], dropout_rate=dropout_rate)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(n_units, activation="relu")
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inp):
        # output dim: [batch_size*n_steps, tonal_size, segment_width]
        inp_reshape = tf.reshape(inp, shape=[-1, self.freq_size, self.segment_width])

        # output dim: [batch_size*n_steps, segment_width, tonal_size]
        inp_permute = tf.transpose(a=inp_reshape, perm=[0, 2, 1])
        inp_permute += positional_encoding(
            batch_size=shape_list(inp_permute)[0], timesteps=self.segment_width, n_units=self.freq_size
        ) * 0.01 + 0.01

        attn_output = self.attn_layer(q=inp_permute, k=inp_permute, v=inp_permute)
        forward_output = self.feed_forward(attn_output)

        # restore shape
        outputs = tf.transpose(a=forward_output, perm=[0, 2, 1])
        outputs = tf.reshape(outputs, shape=[-1, self.n_steps, self.freq_size * self.segment_width])

        outputs = self.dropout(outputs)
        outputs = self.dense(outputs)
        return self.layer_norm(outputs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_steps": self.n_steps,
                "n_units": self.n_units,
                "dropout_rate": self.dropout_rate,
                "freq_size": self.freq_size,
                "segment_width": self.segment_width
            }
        )
        return config


class EncodeSegmentFrequency(tf.keras.layers.Layer):
    """Encode feature along the frequency axis.

    Parameters
    ----------
    n_units: int
        Output embedding size.
    n_steps: int
        Time length of the feature.
    segment_width: int
        Context width of each frame. Nearby frames will be concatenated to the feature axis.
        Default to 21, which means past 10 frames and future 10 frames will be concatenated
        to the current frame, resulting a feature dimenstion of *segment_width x freq_size*.
    freq_size: int
        Feature size of the input representation.
    dropout_rate: float
        Dropout rate of all dropout layers.
    """
    def __init__(self, n_units=512, dropout_rate=0, n_steps=100, freq_size=24, segment_width=21):
        super().__init__()

        self.freq_size = freq_size
        self.segment_width = segment_width
        self.n_steps = n_steps
        self.n_units = n_units
        self.dropout_rate = dropout_rate

        self.attn_layer = MultiHeadAttention(
            n_units=segment_width,
            n_heads=1,
            activation_func="relu",
            relative_position=False,
            max_dist=4,
            dropout_rate=dropout_rate
        )
        self.feed_forward = FeedForward(n_units=[segment_width * 4, segment_width], dropout_rate=dropout_rate)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.out_dense = tf.keras.layers.Dense(n_units, activation="relu")
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inp):
        inp_reshape = tf.reshape(inp, [-1, self.freq_size, self.segment_width])
        inp_reshape += positional_encoding(
            batch_size=shape_list(inp_reshape)[0], timesteps=self.freq_size, n_units=self.segment_width
        ) * 0.01 + 0.01

        attn_output = self.attn_layer(q=inp_reshape, k=inp_reshape, v=inp_reshape)
        forward_output = self.feed_forward(attn_output)

        # restore shape
        outputs = tf.reshape(forward_output, shape=[-1, self.n_steps, self.freq_size * self.segment_width])

        outputs = self.dropout(outputs)
        outputs = self.out_dense(outputs)
        return self.layer_norm(outputs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_steps": self.n_steps,
                "n_units": self.n_units,
                "dropout_rate": self.dropout_rate,
                "freq_size": self.freq_size,
                "segment_width": self.segment_width
            }
        )
        return config


def chord_block_compression(hidden_states, chord_changes):
    block_ids = tf.cumsum(chord_changes, axis=1)
    modify_ids = lambda x: tf.cond(pred=tf.equal(x[0], 0), true_fn=lambda: x, false_fn=lambda: x - 1)
    block_ids = tf.map_fn(modify_ids, block_ids)

    num_blocks = tf.reduce_max(input_tensor=block_ids, axis=1) + 1
    max_steps = tf.reduce_max(input_tensor=num_blocks)

    segment_mean_pad = lambda x: tf.pad(  # pylint: disable=E1123,E1120
        tensor=tf.math.segment_mean(data=x[0], segment_ids=x[1]),
        paddings=tf.convert_to_tensor([[0, max_steps - x[2]], [0, 0]])
    )
    chord_blocks = tf.map_fn(segment_mean_pad, (hidden_states, block_ids, num_blocks), dtype=tf.float32)
    return chord_blocks, block_ids


def chord_block_decompression(compressed_seq, block_ids):
    gather_chords = lambda x: tf.gather(params=x[0], indices=x[1])  # pylint: disable=E1120
    return tf.map_fn(gather_chords, (compressed_seq, block_ids), dtype=compressed_seq.dtype)


def binary_round(inp, cast_to_int=False):
    graph = tf.compat.v1.get_default_graph()
    with ops.name_scope("BinaryRound") as name:
        if cast_to_int:
            with graph.gradient_override_map({"Round": "Identity", "Cast": "Identity"}):
                return tf.cast(tf.round(inp), tf.int32, name=name)
        else:
            with graph.gradient_override_map({"Round": "Identity"}):
                return tf.round(inp, name=name)


class Encoder(tf.keras.layers.Layer):
    """Encoder layer of the transformer model.

    Parameters
    ----------
    num_attn_blocks:
        Number of attention blocks.
    n_steps: int
        Time length of the feature.
    enc_input_emb_size: int
        Embedding size of the encoder's input.
    segment_width: int
        Context width of each frame. Nearby frames will be concatenated to the feature axis.
        Default to 21, which means past 10 frames and future 10 frames will be concatenated
        to the current frame, resulting a feature dimenstion of *segment_width x freq_size*.
    freq_size: int
        Feature size of the input representation.
    dropout_rate: float
        Dropout rate of all the dropout layers.
    **kwargs:
        Other keyword parameters that will be passed to initialize keras.layers.Layer.
    """
    def __init__(
        self,
        dropout_rate=0,
        num_attn_blocks=2,
        n_steps=100,
        enc_input_emb_size=512,
        freq_size=24,
        segment_width=21,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_steps = n_steps
        self.num_attn_blocks = num_attn_blocks
        self.enc_input_emb_size = enc_input_emb_size
        self.dropout_rate = dropout_rate
        self.freq_size = freq_size
        self.segment_width = segment_width

        self.layer_weights = tf.Variable(initial_value=tf.zeros(num_attn_blocks), trainable=True)
        self.encode_segment_time = EncodeSegmentTime(
            n_units=enc_input_emb_size,
            dropout_rate=dropout_rate,
            n_steps=n_steps,
            freq_size=freq_size,
            segment_width=segment_width
        )
        self.attn_layers = [
            MultiHeadAttention(
                n_units=enc_input_emb_size, n_heads=8, max_dist=16, dropout_rate=dropout_rate
            )
            for _ in range(num_attn_blocks)
        ]
        self.ff_layers = [
            FeedForward(n_units=[enc_input_emb_size * 4, enc_input_emb_size], dropout_rate=dropout_rate)
            for _ in range(num_attn_blocks)
        ]
        self.logit_dense = tf.keras.layers.Dense(1)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inp, slope=1):
        segment_encodings = self.encode_segment_time(inp)
        segment_encodings += positional_encoding(
            batch_size=shape_list(segment_encodings)[0], timesteps=self.n_steps, n_units=self.enc_input_emb_size
        )
        segment_encodings = self.dropout(segment_encodings)

        weight = tf.nn.softmax(self.layer_weights)
        weighted_hidden_enc = tf.zeros(shape=shape_list(segment_encodings))
        for idx, (attn_layer, feed_forward) in enumerate(zip(self.attn_layers, self.ff_layers)):
            segment_encodings = attn_layer(q=segment_encodings, k=segment_encodings, v=segment_encodings)
            segment_encodings = feed_forward(segment_encodings)
            weighted_hidden_enc += weight[idx] * segment_encodings

        chord_change_logits = tf.squeeze(self.logit_dense(weighted_hidden_enc))
        chord_change_prob = tf.sigmoid(slope * chord_change_logits)
        chord_change_pred = binary_round(chord_change_prob, cast_to_int=True)

        return weighted_hidden_enc, chord_change_logits, chord_change_pred

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_steps": self.n_steps,
                "enc_input_emb_size": self.enc_input_emb_size,
                "num_attn_blocks": self.num_attn_blocks,
                "dropout_rate": self.dropout_rate,
                "freq_size": self.freq_size,
                "segment_width": self.segment_width
            }
        )
        return config


class Decoder(tf.keras.layers.Layer):
    """Decoder layer of the transformer model.

    Parameters
    ----------
    out_classes: int
        Number of output classes. Currently supports 26 types of chords.
    num_attn_blocks:
        Number of attention blocks.
    n_steps: int
        Time length of the feature.
    dec_input_emb_size: int
        Embedding size of the decoder's input.
    segment_width: int
        Context width of each frame. Nearby frames will be concatenated to the feature axis.
        Default to 21, which means past 10 frames and future 10 frames will be concatenated
        to the current frame, resulting a feature dimenstion of *segment_width x freq_size*.
    freq_size: int
        Feature size of the input representation.
    dropout_rate: float
        Dropout rate of all the dropout layers.
    **kwargs:
        Other keyword parameters that will be passed to initialize keras.layers.Layer.
    """
    def __init__(
        self,
        out_classes=26,
        dropout_rate=0,
        num_attn_blocks=2,
        n_steps=100,
        dec_input_emb_size=512,
        freq_size=24,
        segment_width=21,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_steps = n_steps
        self.dec_input_emb_size = dec_input_emb_size
        self.num_attn_blocks = num_attn_blocks
        self.out_classes = out_classes
        self.dropout_rate = dropout_rate
        self.freq_size = freq_size
        self.segment_width = segment_width

        self.encode_segment_frequency = EncodeSegmentFrequency(
            n_units=dec_input_emb_size,
            dropout_rate=dropout_rate,
            n_steps=n_steps,
            freq_size=freq_size,
            segment_width=segment_width
        )
        self.attn_layers_1 = [
            MultiHeadAttention(
                n_units=dec_input_emb_size,
                n_heads=8,
                relative_position=True,
                max_dist=16,
                self_mask=False,
                dropout_rate=dropout_rate
            )
            for _ in range(num_attn_blocks)
        ]
        self.attn_layers_2 = [
            MultiHeadAttention(
                n_units=dec_input_emb_size,
                n_heads=8,
                relative_position=False,
                max_dist=16,
                self_mask=False,
                dropout_rate=dropout_rate
            )
            for _ in range(num_attn_blocks)
        ]
        self.ff_layers = [
            FeedForward(n_units=[dec_input_emb_size * 4, dec_input_emb_size], dropout_rate=dropout_rate)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.out_dense = tf.keras.layers.Dense(out_classes)

    def call(self, inp, encoder_input_emb, chord_change_pred):
        segment_encodings = self.encode_segment_frequency(inp)
        segment_encodings_blocked, block_ids = chord_block_compression(segment_encodings, chord_change_pred)
        segment_encodings_blocked = chord_block_decompression(segment_encodings_blocked, block_ids)
        segment_encodings_blocked.set_shape([None, self.n_steps, self.dec_input_emb_size])

        decoder_inputs = segment_encodings + segment_encodings_blocked + encoder_input_emb
        decoder_inputs += positional_encoding(
            batch_size=shape_list(decoder_inputs)[0], timesteps=self.n_steps, n_units=self.dec_input_emb_size
        )

        decoder_inputs_drop = self.dropout(decoder_inputs)
        layer_weights = tf.nn.softmax(tf.zeros((self.num_attn_blocks)))
        weighted_hiddens_dec = tf.zeros(shape=shape_list(segment_encodings))
        layer_stack = zip(self.attn_layers_1, self.attn_layers_2, self.ff_layers)
        for idx, (attn_1, attn_2, feed_forward) in enumerate(layer_stack):
            decoder_inputs_drop = attn_1(q=decoder_inputs_drop, k=decoder_inputs_drop, v=decoder_inputs_drop)
            decoder_inputs_drop = attn_2(q=decoder_inputs_drop, k=encoder_input_emb, v=encoder_input_emb)
            decoder_inputs_drop = feed_forward(decoder_inputs_drop)
            weighted_hiddens_dec += layer_weights[idx] * decoder_inputs_drop

        logits = self.out_dense(weighted_hiddens_dec)
        chord_pred = tf.argmax(input=logits, axis=-1, output_type=tf.int32)
        return logits, chord_pred

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_steps": self.n_steps,
                "dec_input_emb_size": self.dec_input_emb_size,
                "num_attn_blocks": self.num_attn_blocks,
                "out_classes": self.out_classes,
                "dropout_rate": self.dropout_rate,
                "freq_size": self.freq_size,
                "segment_width": self.segment_width
            }
        )
        return config


class ChordModel(tf.keras.Model):  # pylint: disable=R0901
    """Chord model in written in keras.

    Keras model of ``chord`` submodule. The original implementation is written in
    tensorflow 1.11 and can be found `here <https://github.com/Tsung-Ping/Harmony-Transformer>`_.

    The model also implements the custom training/test step due to the specialized loss
    computation.

    Parameters
    ----------
    num_enc_attn_blocks: int
        Number of attention blocks in the encoder.
    num_dec_attn_blocks: int
        Number of attention blocks in the decoder.
    segment_width: int
        Context width of each frame. Nearby frames will be concatenated to the feature axis.
        Default to 21, which means past 10 frames and future 10 frames will be concatenated
        to the current frame, resulting a feature dimenstion of *segment_width x freq_size*.
    freq_size: int
        Feature size of the input representation.
    out_classes: int
        Number of output classes. Currently supports 26 types of chords.
    n_steps: int
        Time length of the feature.
    enc_input_emb_size: int
        Embedding size of the encoder's input.
    dec_input_emb_size: int
        Embedding size of the decoder's input.
    dropout_rate: float
        Dropout rate of all the dropout layers.
    annealing_rate: float
        Rate of modifying the slope value for each epoch.
    **kwargs:
        Other keyword parameters that will be passed to initialize the keras.Model.

    See Also
    --------
    omnizart.chord.app.chord_loss_func:
        The customized loss computation function.
    """
    def __init__(
        self,
        num_enc_attn_blocks=2,
        num_dec_attn_blocks=2,
        segment_width=21,
        freq_size=24,
        out_classes=26,
        n_steps=100,
        enc_input_emb_size=512,
        dec_input_emb_size=512,
        dropout_rate=0,
        annealing_rate=1.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.segment_width = segment_width
        self.freq_size = freq_size
        self.out_classes = out_classes
        self.n_steps = n_steps
        self.enc_input_emb_size = enc_input_emb_size
        self.dec_input_emb_size = dec_input_emb_size
        self.dropout_rate = dropout_rate
        self.annealing_rate = annealing_rate

        self.slope = 1
        self.loss_func_name = "chord_loss_func"

        self.encoder = Encoder(
            num_attn_blocks=num_enc_attn_blocks,
            dropout_rate=dropout_rate,
            n_steps=n_steps,
            enc_input_emb_size=enc_input_emb_size,
            freq_size=freq_size,
            segment_width=segment_width
        )
        self.decoder = Decoder(
            num_attn_blocks=num_dec_attn_blocks,
            out_classes=out_classes,
            dropout_rate=dropout_rate,
            n_steps=n_steps,
            dec_input_emb_size=dec_input_emb_size,
            freq_size=freq_size,
            segment_width=segment_width
        )

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, feature):
        encoder_input_emb, chord_change_logits, chord_change_pred = self.encoder(feature, slope=self.slope)
        logits, chord_pred = self.decoder(feature, encoder_input_emb, chord_change_pred)
        return chord_pred, chord_change_pred, logits, chord_change_logits

    def step_in_slope(self):
        self.slope *= self.annealing_rate

    def train_step(self, data):
        # Input feature: (60, 100, 504)
        # Chord change: (60, 100)
        # Chord: (60, 100)
        # Slope: 1.0
        feature, (gt_chord, gt_chord_change) = data

        with tf.GradientTape() as tape:
            chord_pred, chord_change_pred, logits, chord_change_logits = self(feature)

            if self.loss_func_name in self.loss.__name__:
                loss = self.loss(gt_chord, gt_chord_change, logits, chord_change_logits)
                trainable_vars = self.trainable_variables
                loss_l2 = 2e-4 * tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars if "bias" not in var.name])
                loss += loss_l2
            else:
                loss_c = self.compiled_loss(gt_chord, chord_pred)
                loss_cc = self.compiled_loss(gt_chord_change, chord_change_pred)
                loss = loss_c + loss_cc

        # Compute gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update the metrics
        self.compiled_metrics.update_state(gt_chord, chord_pred)
        self.loss_tracker.update_state(loss)
        result = {m.name: m.result() for m in self.metrics}
        result.update({"loss": self.loss_tracker.result()})
        return result

    def test_step(self, data):
        feature, (gt_chord, gt_chord_change) = data
        chord_pred, chord_change_pred, logits, chord_change_logits = self(feature)

        if self.loss_func_name in self.loss.__name__:
            loss = self.loss(gt_chord, gt_chord_change, logits, chord_change_logits)
            trainable_vars = self.trainable_variables
            loss_l2 = 2e-4 * tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars if "bias" not in var.name])
            loss += loss_l2
        else:
            loss_c = self.compiled_loss(gt_chord, chord_pred)
            loss_cc = self.compiled_loss(gt_chord_change, chord_change_pred)
            loss = loss_c + loss_cc

        # Update the metrics
        self.compiled_metrics.update_state(gt_chord, chord_pred)
        self.loss_tracker.update_state(loss)
        result = {m.name: m.result() for m in self.metrics}
        result.update({"loss": self.loss_tracker.result()})
        return result

    def get_config(self):
        config = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "loss_tracker": self.loss_tracker
        }
        return config


class ReduceSlope(tf.keras.callbacks.Callback):
    """Custom keras callback for reducing slope value after each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        self.model.step_in_slope()


if __name__ == "__main__":
    model = ChordModel()
    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy())
    output = model(tf.zeros((16, 60, 100, 504)))
