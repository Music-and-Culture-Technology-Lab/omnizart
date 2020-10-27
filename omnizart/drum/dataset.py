import glob

import h5py
import numpy as np
import tensorflow as tf

from omnizart.utils import get_logger


logger = get_logger("Drum Dataset")


class FeatureLoader:
    """Feature loader of Pop dataset.

    The iterator makes sure that every sample in the dataset will be used.
    """
    def __init__(self, feature_folder=None, feature_files=None, num_samples=100, mini_beat_per_seg=4):
        if feature_files is None:
            assert feature_folder is not None
            self.hdf_files = glob.glob(f"{feature_folder}/*.hdf")
        else:
            self.hdf_files = feature_files

        if len(self.hdf_files) == 0:
            logger.warn("Warning! No feature files found in the given path")

        self.hdf_refs = {hdf: h5py.File(hdf, "r") for hdf in self.hdf_files}
        self.num_samples = num_samples
        self.mini_beat_per_seg = mini_beat_per_seg

        # Initialize indexes index-to-file mapping to ensure all samples
        # will be visited during training.
        length_map = {hdf: len(hdf) for hdf in self.hdf_refs}
        self.total_length = sum(length_map.values())
        self.start_idxs = np.arange(self.total_length, step=self.mini_beat_per_seg)
        self.idx_to_hdf_map = {}
        cur_len = 0
        cur_iid = 0
        for hdf, length in length_map.items():
            end_iid = length // self.mini_beat_per_seg
            for iid in range(cur_iid, cur_iid+end_iid):  # noqa: E226
                start_idx = self.start_idxs[iid]
                self.idx_to_hdf_map[start_idx] = (hdf, start_idx - cur_len)
            cur_len += end_iid * self.mini_beat_per_seg
            cur_iid += end_iid
        diff = len(self.start_idxs) - len(self.idx_to_hdf_map)
        self.cut_idx = int(np.ceil(diff / self.mini_beat_per_seg))
        self.start_idxs = self.start_idxs[:-self.cut_idx]
        np.random.shuffle(self.start_idxs)

        logger.info("Total samples: %s", len(self.start_idxs))

    def __iter__(self):
        for _ in range(self.num_samples):
            if len(self.start_idxs) == 0:
                # Shuffle the indexes after visiting all the samples in the dataset.
                self.start_idxs = np.arange(self.total_length, step=self.mini_beat_per_seg)
                self.start_idxs = self.start_idxs[:-self.cut_idx]
                np.random.shuffle(self.start_idxs)

            start_idx = self.start_idxs[0]
            self.start_idxs = np.delete(self.start_idxs, 0)
            hdf, slice_idx = self.idx_to_hdf_map[start_idx]
            hdf_ref = self.hdf_refs[hdf]
            feat = hdf_ref["feature"][slice_idx:slice_idx+self.mini_beat_per_seg]  # noqa: E226
            feature = np.transpose(feat, axes=[1, 2, 0])  # dim: 120 x 120 x mini_beat_per_seg
            label = hdf_ref["label"][slice_idx:slice_idx+self.mini_beat_per_seg]  # noqa: E226
            label = label.transpose()  # dim: 13 x mini_beat_per_seg

            yield feature, label


def get_dataset(feature_folder=None, feature_files=None, batch_size=8, steps=100):
    """Get the dataset instance.

    A quick way to get and setup the dataset instance for training/validation.

    Parameters
    ----------
    feature_folder: Path
        Path to the folder containing extracted feature files (*.hdf).
    feature_files: list[Path]
        List of paths to the feature files with extension `.hdf`. This parameter
        gives the ability to have more control on which files to use.

        Either ``feature_folder`` or ``feature_files`` should be specified.
        If both given, ``feature_files`` will be used.
    batch_size: int
        Size of input batch for each training step.
    steps: int
        Total steps of each epoch.

    Returns
    -------
    dataset: tf.data.Dataset
        A tensorflow dataset instance with optimized setup.
    """
    loader = FeatureLoader(
        feature_folder=feature_folder,
        feature_files=feature_files,
        num_samples=batch_size*steps,  # noqa: E226
    )

    def gen_wrapper():
        for data in loader:
            yield data

    return tf.data.Dataset.from_generator(
        gen_wrapper,
        output_types=(tf.float32, tf.float32),
        output_shapes=([120, 120, 4], [4, 13])) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.experimental.AUTOTUNE)


if __name__ == "__main__":
    feat_files = [
        "/data/omnizart/bootstrap_data/drum/feature/ytd_audio_00002_TRGDXTK128F14584C1.hdf"
    ]
    loader = iter(FeatureLoader(feature_files=feat_files))
