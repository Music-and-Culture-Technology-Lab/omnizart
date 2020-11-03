import glob

import h5py
import numpy as np
import tensorflow as tf

from omnizart.utils import get_logger


logger = get_logger("Chord Dataset")


class FeatureLoader:
    """Feature loader of ``chord`` module.

    Parameters
    ----------
    feature_folder: Path
        Path to the folder that contains the extracted feature files.
    feature_files: list[Path]
        List of paths to the feature files. One of *feature_folder* or *feature_files*
        must be specified.
    num_samples: int
        Total sample number to be yielded.

    Yields
    ------
    feature:
        Input feature for training the model.
    label: tuple
        gt_chord -> Ground-truth chord label.
        gt_chord_change -> Ground-truth chord change label.
    """
    def __init__(self, feature_folder=None, feature_files=None, num_samples=100):
        if feature_files is None:
            assert feature_folder is not None
            self.hdf_files = glob.glob(f"{feature_folder}/*.hdf")
        else:
            self.hdf_files = feature_files

        if len(self.hdf_files) == 0:
            logger.warning("Warning! No feature files found in the given path")

        self.hdf_refs = {hdf: h5py.File(hdf, "r") for hdf in self.hdf_files}
        self.num_samples = num_samples

        length_map = {hdf: len(hdf_ref["chroma"]) for hdf, hdf_ref in self.hdf_refs.items()}
        self.total_length = sum(length_map.values())
        self.start_idxs = np.arange(self.total_length)
        self.idx_to_hdf_map = {}
        cur_len = 0
        for hdf, length in length_map.items():
            for iid in range(cur_len, cur_len + length):
                start_idx = self.start_idxs[iid]
                self.idx_to_hdf_map[start_idx] = (hdf, start_idx - cur_len)
            cur_len += length
        np.random.shuffle(self.start_idxs)

        logger.info("Total samples: %s", len(self.start_idxs))

    def __iter__(self):
        for _ in range(self.num_samples):
            if len(self.start_idxs) == 0:
                # Shuffle the indexes after visiting all the samples in the dataset.
                self.start_idxs = np.arange(self.total_length)
                np.random.shuffle(self.start_idxs)

            start_idx = self.start_idxs[0]
            self.start_idxs = self.start_idxs[1:]  # Remove the first element
            hdf, slice_idx = self.idx_to_hdf_map[start_idx]
            hdf_ref = self.hdf_refs[hdf]
            feat = hdf_ref["chroma"][slice_idx]
            gt_chord = hdf_ref["chord"][slice_idx]
            gt_chord_change = hdf_ref["chord_change"][slice_idx]

            yield feat, (gt_chord, gt_chord_change)


def get_dataset(feature_folder=None, feature_files=None, epochs=10, batch_size=8, steps=100):
    loader = FeatureLoader(
        feature_folder=feature_folder,
        feature_files=feature_files,
        num_samples=epochs*batch_size*steps  # noqa: E226
    )

    def gen_wrapper():
        for data in loader:
            yield data

    return tf.data.Dataset.from_generator(
        gen_wrapper,
        output_types=(tf.float32, (tf.int32, tf.int32)),
        output_shapes=([100, 504], ([100], [100]))) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.experimental.AUTOTUNE)
