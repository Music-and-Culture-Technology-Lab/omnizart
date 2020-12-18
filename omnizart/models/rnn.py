import tensorflow as tf

from omnizart.models.t2t import MultiHeadAttention


def blstm(timesteps=1200, input_dim=178, hidden_dim=25, num_lstm_layers=2):
    inputs = tf.keras.Input(shape=(timesteps, input_dim))

    bl_out = tf.keras.layers.LayerNormalization()(inputs)
    for _ in range(num_lstm_layers):
        bl_out = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        )(bl_out)
    bl_out_flatten = tf.keras.layers.Flatten()(bl_out)

    beat_out = tf.keras.layers.Dense(timesteps, activation="sigmoid")(bl_out_flatten)
    down_beat_out = tf.keras.layers.Dense(timesteps, activation="sigmoid")(bl_out_flatten)

    out = tf.stack([beat_out, down_beat_out], axis=2)
    return tf.keras.Model(inputs=inputs, outputs=out)


def blstm_attn(timesteps=1200, input_dim=178, lstm_hidden_dim=25, num_lstm_layers=2, attn_hidden_dim=256):
    inputs = tf.keras.Input(shape=(timesteps, input_dim))

    bl_out = tf.keras.layers.LayerNormalization()(inputs)
    for _ in range(num_lstm_layers):
        bl_out = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_hidden_dim, return_sequences=True)
        )(bl_out)

    attn_in = tf.keras.layers.Conv1D(attn_hidden_dim, 1)(bl_out)
    attn_out = MultiHeadAttention(
        n_units=attn_hidden_dim, dropout_rate=0.1, relative_position=True, max_dist=800, causal=True
    )(attn_in, attn_in, attn_in)
    attn_out_flatten = tf.keras.layers.Flatten()(attn_out)

    beat_out = tf.keras.layers.Dense(timesteps, activation="sigmoid")(attn_out_flatten)
    down_beat_out = tf.keras.layers.Dense(timesteps, activation="sigmoid")(attn_out_flatten)

    out = tf.stack([beat_out, down_beat_out], axis=2)
    return tf.keras.Model(inputs=inputs, outputs=out)


if __name__ == "__main__":
    inputs = tf.random.normal([3, 1200, 178])
    model = blstm_attn()
    out = model(inputs)
