# pylint: disable=C0103,R0914,R0915

import glob
import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical
from omnizart.models.utils import note_res_downsampling, padding


# Brought from https://github.com/s603122001/Vocal-Melody-Extraction/blob/master/project/train.py
# and adapted to load hdf files.
def generator_audio(feature_folder, batch_size, timesteps, phase='train', percentage_train=0.8):
    hdf_files = glob.glob(f"{feature_folder}/*.hdf")
    if len(hdf_files) == 0:
        print("Warning! No feature files found in the given path")

    X_48 = []
    X_12 = []
    Y = []
    for hdf in hdf_files:
        with h5py.File(hdf, "r") as f:
            X_48.append(np.array(f["feature"]))
            Y.append(np.array(f["label"]))
    X_48 = np.array(X_48)
    Y = np.array(Y)

    # Set chorale_indices
    if phase == 'train':
        chorale_indices = np.arange(int(len(X_48) * percentage_train))
    if phase == 'test':
        chorale_indices = np.arange(int(len(X_48) * percentage_train), len(X_48))
    if phase == 'all':
        chorale_indices = np.arange(int(len(X_48)))

    for clip_ind, clip in enumerate(X_48):
        if clip_ind in chorale_indices:
            new_x = np.array(clip[:, :, 0])

            new_x_12 = note_res_downsampling(new_x)
            new_x_12 = padding(new_x_12, 128, timesteps)
            X_12.append(new_x_12)

            new_x_48 = padding(new_x, 384, timesteps)
            X_48[clip_ind] = new_x_48

            new_y = np.array(Y[clip_ind])
            Y[clip_ind] = padding(new_y, 384, timesteps)
        else:
            X_48[clip_ind] = 0
            X_12.append(0)
            Y[clip_ind] = 0

    features_48 = []
    features_12 = []
    labels = []

    batch = 0

    while True:
        # control the training percentage between datasets(we use 1/2 mir1k and 1/2 medleydb)
        if (np.random.choice(np.arange(10)) < 5 and phase == 'train'):
            chorale_index = np.random.choice(np.arange(48))
        else:
            chorale_index = np.random.choice(chorale_indices)
        chorale = np.array(X_48[chorale_index])
        chorale_length = len(chorale)
        time_index = np.random.randint(0, chorale_length - timesteps)

        feature_48 = (X_48[chorale_index][time_index: time_index + timesteps])
        feature_48 = np.reshape(feature_48, (timesteps, 384, 1))

        feature_12 = (X_12[chorale_index][time_index: time_index + timesteps])
        feature_12 = np.reshape(feature_12, (timesteps, 128, 1))

        label = Y[chorale_index][time_index: time_index + timesteps]
        label = to_categorical(label, num_classes=2)

        features_48.append(feature_48)
        features_12.append(feature_12)
        labels.append(label)

        batch += 1

        # if there is a full batch
        if batch == batch_size:
            next_element = (
                np.array(features_48, dtype=np.float32),
                np.array(features_12, dtype=np.float32),
                np.array(labels, dtype=np.float32),
            )

            yield next_element

            batch = 0

            features_48 = []
            features_12 = []
            labels = []
