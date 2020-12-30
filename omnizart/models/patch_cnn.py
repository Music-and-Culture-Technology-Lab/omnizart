import tensorflow as tf


def patch_cnn_model(patch_size=25):
    inputs = tf.keras.Input(shape=(patch_size, patch_size, 1))

    out = tf.keras.layers.Conv2D(8, 5, activation="relu")(inputs)
    out = tf.keras.layers.Dropout(0.25)(out)
    out = tf.keras.layers.Conv2D(16, 3, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.25)(out)
    out = tf.keras.layers.MaxPooling2D(2, 2)(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(128, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.25)(out)
    out = tf.keras.layers.Dense(64, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.25)(out)
    out = tf.keras.layers.Dense(2, activation="softmax")(out)

    return tf.keras.Model(inputs=inputs, outputs=out)
