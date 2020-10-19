# pylint: disable=W0223,W0102
import glob
import json
import pickle
import random

import h5py
import tensorflow as tf

from omnizart.music.labels import LabelType


class FeatureDataset(tf.data.Dataset):
    """Dataset loader for ``music`` module.

    It's much faster than passing the naive generator to the training loop.
    There are some work-around to dealing with the customized stored label format.
    Due to that all return data and corresponding type need to be defined first,
    meaning no type conversion are allowed while applying the `map` function,
    and thus the work-around is to add one more column to the return data as the
    intermediate data that will not be used in the training loop, but will be
    used in the `map` processing.
    In ``music`` module, the `map` conversion happened in ``get_label_conversion_wrapper``,
    and must be applied, or it will return the wrong data pair.

    Parameters
    ----------
    feature_folder: Path
        Path to the extracted feature files, including `*.hdf` and `*.pickle` pairs,
        which refers to feature and label files, respectively.
    feature_files: list[Path]
        List of path of `*.hdf` feature files. Corresponding label files should also
        under the same folder.
    num_samples: int
        Total number of samples to be yielded.
    timesteps: int
        Time length of the feature.
    channels: list[int]
        Channels to be used for training. Allowed values are [1, 2, 3].

    Yields
    ------
    feature: numpy.ndarray
        Input feature for training.
    label: numpy.ndarray
        Corresponding label.
    label_str: str
        Column of intermediate product. Should not be used in the training.

    Notes
    -----
    One of the parameter, `feature_folder` or `feature_files`, should be specified.
    """
    @staticmethod
    def _generator(feature_folder=None, feature_files=None, num_samples=100, slice_len=128, channels=[1, 3]):
        # sys.stdout.write('\033[2K\033[1G')  # Clear previous output
        # print("Loading data")

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

        half_slice_len = int(round(slice_len / 2))
        for _ in range(num_samples):
            key = random.choice(hdf_keys)
            hdf_ref = hdf_refs[key]
            length = min(len(hdf_ref["feature"]), len(pkls[key]))
            start_idx = half_slice_len
            end_idx = length - half_slice_len
            center_id = random.randint(start_idx, end_idx)
            slice_range = range(center_id-half_slice_len, center_id+half_slice_len)  # noqa: E226

            feature = hdf_refs[key]["feature"][slice_range][:]
            label = pkls[key][slice_range[0]:slice_range[-1]+1]  # noqa: E226
            yield feature[:, :, channels], [], json.dumps(label)

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
    def wrapper(feat, empty, label_str):  # pylint: disable=W0613
        lnpy = label_str.numpy()
        if not isinstance(lnpy, bytes):
            # Batched input
            tensor_list = []
            for l_str in lnpy:
                tensor_list.append(convert_label_str(l_str, conversion_func))
            return feat, tf.stack(tensor_list), tf.convert_to_tensor([""] * len(label_str))
        return feat, convert_label_str(lnpy, conversion_func), label_str, tf.convert_to_tensor([""])
    return tf.py_function(wrapper, inp=(feat, empty, label_str), Tout=(feat.dtype, empty.dtype, label_str.dtype))


class FeatureLoader:
    """Feature loader for training ``music`` model.

    Random samples the feature and label pair in the dataset, does not gaurantee that
    all samples will be used while training.
    """
    def __init__(
        self,
        label_conversion_func,
        feature_folder=None,
        feature_files=None,
        num_samples=100,
        timesteps=128,
        channels=[1, 3]
    ):
        if feature_files is None:
            assert feature_folder is not None
            self.hdf_files = glob.glob(f"{feature_folder}/*.hdf")
        else:
            self.hdf_files = feature_files

        self.conv_func = label_conversion_func
        self.feature_folder = feature_folder
        self.feature_files = feature_files
        self.num_samples = num_samples
        self.timesteps = timesteps
        self.channels = channels

        self.hdf_refs = {}
        self.pkls = {}
        for hdf in self.hdf_files:
            ref = h5py.File(hdf, "r")
            self.hdf_refs[hdf] = ref
            self.pkls[hdf] = pickle.load(open(hdf.replace(".hdf", ".pickle"), "rb"))["label"]
        self.hdf_keys = list(self.hdf_refs.keys())

    def __iter__(self):
        half_slice_len = int(round(self.timesteps / 2))
        for _ in range(self.num_samples):
            key = random.choice(self.hdf_keys)
            length = min(len(self.hdf_refs[key]["feature"]), len(self.pkls[key]))
            start_idx = half_slice_len
            end_idx = length - half_slice_len
            center_id = random.randint(start_idx, end_idx)
            slice_range = range(center_id-half_slice_len, center_id+half_slice_len)  # noqa: E226

            feature = self.hdf_refs[key]["feature"][slice_range][:]
            label = self.pkls[key][slice_range[0]:slice_range[-1]+1]  # noqa: E226
            yield feature[:, :, self.channels], self.conv_func(label)


def get_dataset(
    label_conversion_func,
    feature_folder=None,
    feature_files=None,
    batch_size=8,
    steps=100,
    timesteps=128,
    channels=[1, 3]
):
    """Get the dataset instance.

    Use this function to get the dataset instance and don't initialize
    the dataset instance yourself, since it may lead to unknown behavior
    due to the customized process.

    Parameters
    ----------
    label_conversion_func: callable
        The function that will be used for converting the customized label format
        into numpy array.
    feature_folder: Path
        Path to the extracted feature files, including `*.hdf` and `*.pickle` pairs,
        which refers to feature and label files, respectively.
    feature_files: list[Path]
        List of path of `*.hdf` feature files. Corresponding label files should also
        under the same folder.
    batch_size: int
        Size of input batch for each step.
    steps: int
        Total steps for each epoch.
    timesteps: int
        Time length of the feature.
    channels: list[int]
        Channels to be used for training. Allowed values are [1, 2, 3].
    """
    loader = FeatureLoader(
        label_conversion_func,
        feature_folder=feature_folder,
        feature_files=feature_files,
        num_samples=batch_size*steps,  # noqa: E226
        timesteps=timesteps,
        channels=channels
    )

    def gen_wrapper():
        for data in loader:
            yield data

    return tf.data.Dataset.from_generator(
        gen_wrapper, output_types=(tf.float32, tf.float32)) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.experimental.AUTOTUNE)


if __name__ == "__main__":
    l_type = LabelType("note-stream")
    FEAT_FOLDER = "/host/home/76_pop_rhythm/train_feature/slice"
    # loader = FeatureLoader(feature_folder=FEAT_FOLDER)
    dataset = get_dataset(l_type.get_conversion_func(), feature_folder=FEAT_FOLDER)
