"""Utility functions of processing file I/O."""

import os
import pickle


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
