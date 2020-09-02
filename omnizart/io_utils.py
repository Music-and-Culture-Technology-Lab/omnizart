"""Utility functions of processing file I/O."""

import os
import pickle

import librosa

from omnizart.constants.feature import DOWN_SAMPLE_TO_SAPMLING_RATE


def dump_pickle(data, save_to):
    """Dump data to the given path.

    Parameters
    ----------
    data: python objects
        Data to store. Should be python built-in types like `dict`, `list`, `str`, `int`, etc
    save_to: Path
        The full path to store the pickle file, including file name.
        Will create the directory if the given path doesn't exist.

    """
    base_dir = os.path.dirname(save_to)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    with open(save_to, "wb") as pkl_file:
        pickle.dump(data, pkl_file)


def load_pickle(pickle_file):
    """Load pickle file from the given path

    Read and returns the data from the given pickle file path

    Parameters
    ----------
    pickle_file: Path
        The full path to the pickle file for read

    Returns
    -------
    object
        Python object, could be `dict`, `list`, `str`, etc.
    """
    return pickle.load(open(pickle_file, "rb"))


def load_audio_with_librosa(audio_path, sampling_rate=DOWN_SAMPLE_TO_SAPMLING_RATE):
    """Load audio from the given path with librosa.load
    
    Parameters
    ----------
    audio_path: Path
        Path to the audio.
    sampling_rate: int
        Target sampling rate after loaded.
    
    Returns
    -------
    audio: 1D numpy array
        Raw data of the audio.
    tar_sampling_rate: int
        Sampling rate of the audio. Will be the same as the given ``sampling_rate``.
    """
    # Returns: 1D-array, sampling rate (int)
    return librosa.load(audio_path, mono=True, sr=sampling_rate)
