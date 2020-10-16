import glob

import h5py
import numpy as np
import tensorflow as tf


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

        self.hdf_refs = {hdf: h5py.File(hdf, "r") for hdf in self.hdf_files}
        self.num_samples = num_samples
        self.mini_beat_per_seg = mini_beat_per_seg

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
                self.idx_to_hdf_map[start_idx] = (hdf, start_idx-cur_len)  # noqa: E226
            cur_len += end_iid * self.mini_beat_per_seg
            cur_iid += end_iid
        np.random.shuffle(self.start_idxs)

    def __iter__(self):
        for _ in range(self.num_samples):
            if len(self.start_idxs) == 0:
                self.start_idxs = np.arange(self.total_length, step=self.mini_beat_per_seg)
                np.random.shuffle(self.start_idxs)

            start_idx = self.start_idxs[0]
            np.delete(self.start_idxs, 0)
            hdf, slice_idx = self.idx_to_hdf_map[start_idx]
            hdf_ref = self.hdf_refs[hdf]
            feature = hdf_ref["feature"][slice_idx:slice_idx+self.mini_beat_per_seg]  # noqa: E226
            label = hdf_ref["label"][slice_idx:slice_idx+self.mini_beat_per_seg]  # noqa: E226
            yield feature, label


def get_dataset(feature_folder=None, feature_files=None, batch_size=8, steps=100):
    loader = FeatureLoader(
        feature_folder=feature_folder,
        feature_files=feature_files,
        num_samples=batch_size*steps,  # noqa: E226
    )

    def gen_wrapper():
        for data in loader:
            yield data

    return tf.data.Dataset.from_generator(
        gen_wrapper, output_types=(tf.float32, tf.float32)) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.experimental.AUTOTUNE)
