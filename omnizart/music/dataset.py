import sys
import glob
import json
import h5py
import pickle
import random

import numpy as np
import tensorflow as tf

from omnizart.music.labels import LabelType


class FeatureDataset(tf.data.Dataset):
    @staticmethod
    def _generator(feature_folder=None, feature_files=None,num_samples=100, slice_len=128, channels=[1, 3]):
        sys.stdout.write('\033[2K\033[1G')  # Clear previous output
        print("Loading data")

        if feature_files is None: 
            assert feature_folder is not None
            hdf_files = glob.glob(f"{feature_folder}/*.hdf")
        else:
            hdf_files = feature_files
        hdf_refs = {}
        pkls = {}
        for hdf in hdf_files:
            ref = h5py.File(hdf, "r")
            hdf_refs[hdf] = ref
            pkls[hdf] = pickle.load(open(hdf.replace(".hdf", ".pickle"), "rb"))["label"]
        hdf_keys = list(hdf_refs.keys())

        half_slice_len = int(round(slice_len/2))
        for _ in range(num_samples):
            key = random.choice(hdf_keys)
            hdf_ref = hdf_refs[key]
            length = min(len(hdf_ref["feature"]), len(pkls[key]))
            start_idx = half_slice_len
            end_idx = length - half_slice_len
            center_id = random.randint(start_idx, end_idx)
            slice_range = range(center_id-half_slice_len, center_id+half_slice_len)

            feature = hdf_refs[key]["feature"][slice_range][:]
            label = pkls[key][slice_range[0]:slice_range[-1]+1]
            yield feature[:,:,channels], [], json.dumps(label)

    def __new__(cls, feature_folder=None, feature_files=None, num_samples=8, timesteps=128, channels=[1, 3]):
        no_arg_generator = lambda: cls._generator(
            feature_folder=feature_folder,
            feature_files=feature_files,
            num_samples=num_samples,
            slice_len=timesteps,
            channels=channels
        )

        return tf.data.Dataset.from_generator(
            no_arg_generator,
            output_types=(tf.float32, tf.float32, tf.string),
        )


def convert_label_str(label_str, conversion_func):
    label = json.loads(label_str)
    label_npy = conversion_func(label)
    return tf.convert_to_tensor(label_npy.astype("float32"))


def get_label_conversion_wrapper(feat, empty, label_str, conversion_func):
    def wrapper(feat, empty, label_str):
        lnpy = label_str.numpy()
        if not isinstance(lnpy, bytes):
            # Batched input
            tensor_list = []
            for l_str in lnpy:
                tensor_list.append(convert_label_str(l_str, conversion_func))
            return feat, tf.stack(tensor_list), tf.convert_to_tensor([""]*len(label_str))
        return feat, convert_label_str(lnpy, conversion_func), label_str, tf.convert_to_tensor([""])
    return tf.py_function(wrapper, inp=(feat, empty, label_str), Tout=(feat.dtype, empty.dtype, label_str.dtype))


def get_dataset(
    label_conversion_func,
    feature_folder=None,
    feature_files=None,
    batch_size=8,
    steps=100,
    timesteps=128,
    channels=[1, 3]
):
    assert (feature_folder is not None) or (feature_files is not None)
    return FeatureDataset(
            feature_folder=feature_folder,
            feature_files=feature_files,
            num_samples=batch_size*steps,
            timesteps=timesteps,
            channels=channels
        ) \
        .batch(batch_size, drop_remainder=True) \
        .map(
            lambda feat, empty, label_str: get_label_conversion_wrapper(feat, empty, label_str, label_conversion_func),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .cache() \
        .prefetch(tf.data.experimental.AUTOTUNE)


if __name__ == "__main__":
    l_type = LabelType("pop-note-stream")
    dataset = get_dataset(l_type.get_conversion_func(), feature_folder="/data/omnizart/tf_dataset_experiment/feature")
