import os
import csv
import pickle

import yaml
import librosa

from omnizart.utils import ensure_path_exists, LazyLoader, get_logger


# Lazy load the Spleeter pacakge for avoiding pulling large dependencies
# and boosting the import speed.
adapter = LazyLoader("adapter", globals(), "spleeter.audio.adapter")
logger = get_logger("IO")


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
    ensure_path_exists(base_dir)
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


def load_audio(audio_path, sampling_rate=44100, mono=True):
    """Load audio with spleeter.

    A much faster and general approach for loading audio comparing to use librosa.
    This function also allows to read .mp3 files.

    Parameters
    ----------
    audio_path: Path
        Path to the audio.
    sampling_rate: int
        Target sampling rate after loaded.
    mono: bool
        Wether to transform the audio into monophonic channel.

    Returns
    -------
    audio: 1D numpy array
        Raw data of the audio.
    fs: int
        Sampling rate of the audio. Will be the same as the given ``sampling_rate``.
    """
    try:
        audio_loader = adapter.AudioAdapter.default()
        audio, fs = audio_loader.load(audio_path, sample_rate=sampling_rate)
        if mono:
            audio = librosa.to_mono(audio.squeeze().T)

    except (ImportError, adapter.SpleeterError) as error:
        logger.warning(
            "Failed to load audio with Spleeter due to '%s'. Continue to use Librosa.", str(error)
        )
        audio, fs = load_audio_with_librosa(audio_path, sampling_rate=sampling_rate, mono=mono)
        if not mono:
            audio = audio.T

    return audio, fs


def load_audio_with_librosa(audio_path, sampling_rate=44100, mono=True):
    """Load audio from the given path with librosa.load

    Parameters
    ----------
    audio_path: Path
        Path to the audio.
    sampling_rate: int
        Target sampling rate after loaded.
    mono: bool
        Wether to transform the audio into monophonic channel.

    Returns
    -------
    audio: 1D numpy array
        Raw data of the audio.
    fs: int
        Sampling rate of the audio. Will be the same as the given ``sampling_rate``.
    """
    return librosa.load(audio_path, mono=mono, sr=sampling_rate)


def load_yaml(yaml_path):
    return yaml.load(open(yaml_path, "r"), Loader=yaml.Loader)


def write_yaml(json_obj, output_path, dump=True):
    # If dump is false, then the json_obj should be yaml string already.
    out_str = yaml.dump(json_obj) if dump else json_obj
    open(output_path, "w").write(out_str)


def write_agg_f0_results(agg_f0, output_path):
    """Write out aggregated F0 information as a CSV file.

    Parameters
    ----------
    agg_f0: list[dict]
        List of aggregated F0 information.
    output_path: Path
        Path for output the CSV file. Should contain the file name.

    See Also
    --------
    omnizart.utils.aggregate_f0_info:
        The function for generating the aggregated F0 information.
    """
    with open(output_path, "w") as out:
        writer = csv.DictWriter(out, fieldnames=["start_time", "end_time", "frequency"])
        writer.writeheader()
        writer.writerows(agg_f0)
