"""Information about directory structure of each dataset."""
# pylint: disable=C0112
import os
import glob
from shutil import copy

from omnizart.utils import ensure_path_exists, get_logger
from omnizart.remote import download_large_file_from_google_drive


logger = get_logger("Constant Datasets")


def _get_file_list(dataset_path, dirs, ext):
    files = []
    for _dir in dirs:
        files += glob.glob(os.path.join(dataset_path, _dir, "*" + ext))
    return files


class BaseStructure:
    """Defines the necessary attributes and common functions for each sub-dataset structure class.

    All sub-dataset structure class should inherit this base class to ensure
    the necessary attributes and methods are overriden.
    """

    #: The URL for downloading the dataset.
    url = None

    #: The extension of ground-truth files (e.g. .mid, .csv).
    label_ext = None

    train_wavs = None
    """Records folders that contains trainig wav files.

    The path to sub-folders should be the relative path to root folder of the dataset.

    Examples
    --------
    Assume the structure of the dataset looks like following

    |  maps
    |  ├── MAPS_AkPnCGdD_2
    |  │   └── AkPnCGdD
    |  └── MAPS_AkPnBcht_2
    |      └── AkPnBcht

    Where ``AkPnCGdD`` and ``AkPnBcht`` are the folders store the wav files.
    The function should return a list like:

    |  >>> ['MAPS_AkPnCGdD_2/AkPnCGdD', 'MAPS_AkPnBcht_2/AkPnBcht']
    """

    test_wavs = None
    """Records folders that contains testing wav files.

    Same as what `train_wavs` does, but for testing wav files.
    """

    train_labels = None
    """Records folders that contains training labels.

    Similar to the `train_wavs` function, records information of where the corresponding
    ground-truth files are stored.
    """

    test_labels = None
    """Records folders that contains testing labels.

    Similar to the `train_labels` function, records information of where the corresponding
    ground-truth files are stored.
    """

    @classmethod
    def _get_data_pair(cls, wavs, labels):
        label_path_mapping = {os.path.basename(label): label for label in labels}

        pair = []
        for wav in wavs:
            basename = os.path.basename(wav)
            label_name = cls._name_transform(basename).replace(".wav", cls.label_ext)
            label_path = label_path_mapping[label_name]
            assert os.path.exists(label_path)
            pair.append((wav, label_path))

        return pair

    @classmethod
    def get_train_data_pair(cls, dataset_path):
        """Get pair of training file and the coressponding label file path."""
        return cls._get_data_pair(cls.get_train_wavs(dataset_path), cls.get_train_labels(dataset_path))

    @classmethod
    def get_test_data_pair(cls, dataset_path):
        """Get pair of testing file and the coressponding label file path."""
        return cls._get_data_pair(cls.get_test_wavs(dataset_path), cls.get_test_labels(dataset_path))

    @classmethod
    def _name_transform(cls, basename):
        return basename

    @classmethod
    def get_train_wavs(cls, dataset_path):
        """Get list of complete train wav paths"""
        return _get_file_list(dataset_path, cls.train_wavs, ".wav")

    @classmethod
    def get_test_wavs(cls, dataset_path):
        """Get list of complete test wav paths"""
        return _get_file_list(dataset_path, cls.test_wavs, ".wav")

    @classmethod
    def get_train_labels(cls, dataset_path):
        """Get list of complete train label paths"""
        return _get_file_list(dataset_path, cls.train_labels, cls.label_ext)

    @classmethod
    def get_test_labels(cls, dataset_path):
        """Get list of complete test label paths"""
        return _get_file_list(dataset_path, cls.test_labels, cls.label_ext)

    @classmethod
    def download(cls, save_path="./"):
        """Download the dataset.

        After download the compressed (zipped) dataset file, the function will automatically
        decompress it and delete the original zipped file.

        You can apply some post process after download by overriding the function ``_post_download``.
        The _post_download function receives a single ``dataset_path`` as the parameter,
        and you can do anything to the dataset such as re-organize the directory structure,
        or filter out some files.
        """
        ensure_path_exists(save_path)
        save_name = cls.__name__.replace("Structure", "") + ".zip"
        dataset_path = download_large_file_from_google_drive(
            cls.url, save_path=save_path, save_name=save_name, unzip=True
        )
        os.remove(save_name)
        cls._post_download(dataset_path=dataset_path)

    @classmethod
    def _post_download(cls, dataset_path):
        pass


class MapsStructure(BaseStructure):
    """Structure of MAPS dataset"""

    #: Label extension
    label_ext = ".txt"

    #: Folder to train wavs
    train_wavs = [
        "MAPS_AkPnBcht_2/AkPnBcht/MUS",
        "MAPS_AkPnBsdf_2/AkPnBsdf/MUS",
        "MAPS_AkPnStgb_2/AkPnStgb/MUS",
        "MAPS_AkPnCGdD_2/AkPnCGdD/MUS",
        "MAPS_SptkBGCl_2/SptKBGCl/MUS",
        "MAPS_StbgTGd2_2/StbgTGd2/MUS",
    ]

    #: Folder to test wavs
    test_wavs = ["MAPS_ENSTDkAm_2/ENSTDkAm/MUS", "MAPS_ENSTDkCl_2/ENSTDkCl/MUS"]

    #: Folder to train labels
    train_labels = train_wavs

    #: Folder to test labels
    test_labels = test_wavs


class MusicNetStructure(BaseStructure):
    """Structure of MusicNet dataset"""

    #: Dataset URL
    url = "https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz"

    #: Label extension
    label_ext = ".csv"

    #: Folder to train wavs
    train_wavs = ["train_data"]

    #: Folder to test wavs
    test_wavs = ["test_data"]

    #: Folder to train labels
    train_labels = ["train_labels"]

    #: Folder to test labels
    test_labels = ["test_labels"]


class MaestroStructure(BaseStructure):
    """Structure of Maestro dataset"""

    #: Dataset URL
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip"

    #: Label extension
    label_ext = ".midi"

    #: Folder to train wavs
    train_wavs = ["2004", "2006", "2008", "2009", "2011", "2013", "2014", "2015", "2017"]

    #: Folder to test wavs
    test_wavs = ["2018"]

    #: Folder to train labels
    train_labels = train_wavs

    #: Folder to test labels
    test_labels = test_wavs


class PopStructure(BaseStructure):
    """Structure of Pop dataset"""

    #: Label extension
    label_ext = ".mid"

    #: Folder to train wavs
    train_wavs = [
        "01_ytd_audio/dist0p00",
        "01_ytd_audio/dist0p10",
        "01_ytd_audio/dist0p20",
        "01_ytd_audio/dist0p40",
        "01_ytd_audio/dist0p50",
        "01_ytd_audio/dist0p60",
    ]

    #: Folder to test wavs
    test_wavs = ["01_ytd_audio/dist0p30"]

    #: Folder to train labels
    train_labels = [
        "05_align_mid/dist0p00",
        "05_align_mid/dist0p10",
        "05_align_mid/dist0p20",
        "05_align_mid/dist0p40",
        "05_align_mid/dist0p50",
        "05_align_mid/dist0p60",
    ]

    #: Folder to test labels
    test_labels = ["05_align_mid/dist0p30"]


class ExtSuStructure(BaseStructure):
    """Structure of Extended-Su dataset"""

    #: Dataset URL
    url = "https://drive.google.com/uc?export=download&id=1Miw9G2O1Y8g253RQ2uQ4udM5XMKB-9p8"

    #: Label extension
    label_ext = ".mid"

    #: Folder to train wavs
    train_wavs = []

    #: Folder to test wavs
    test_wavs = [
        "1 Tchaikovsky", "2 schumann", "3 beethoven", "5 Mozart",
        "PQ01_Dvorak", "PQ02_Elgar", "PQ03_Farranc", "PQ04_Frank", "PQ05_Hummel",
        "PQ06_Schostakovich", "PQ07_Schubert", "PQ08_Schubert",
        "SQ01_Beethoven", "SQ02_Janacek", "SQ03_Schubert", "SQ04_Janacek",
        "SQ04_Ravel", "SQ05_Mozart", "SQ07_Haydn", "SQ08_Dvorak", "SQ09_Ravel",
        "SY06_Mahler",
        "VS01_Schumann", "VS02_Brahms", "VS03_Debussy", "VS04_Franck", "VS05_Mozart",
        "WQ01_Nielsen", "WQ02_Schoenberg", "WQ03_Cambini", "WQ04_Danzi",
    ]

    #: Folder to train labels
    train_labels = train_wavs

    #: Folder to test labels
    test_labels = test_wavs


class McGillBillBoard(BaseStructure):
    """Constant settings of McGill BillBoard dataset."""

    #: Path to the feature folder relative to dataset
    feature_folder = "./McGill-Billboard-Features"

    #: Path to the label folder relative to dataset
    label_folder = "./McGill-Billboard-MIREX"

    #: Name of feature files
    feature_file_name = "bothchroma.csv"

    #: Name of label files
    label_file_name = "majmin.lab"

    #: Path to the index file relative the dataset
    index_file_path = "./billboard-2.0-index.csv"

    #: Split ID of train/val set.
    train_test_split_id = 1000

    #: Dataset URL
    url = "https://drive.google.com/uc?export=download&id=1_K_Fof4zt1IQvs1aDmf-5wY0wHqgcPlC"


class BeethovenSonatasStructure(BaseStructure):
    """Constant settings of BPS-FH dataset"""

    #: Hosted dataset download url.
    url = "https://drive.google.com/uc?export=download&id=1nYq2FB5LQfYJoXyYZl3XcklpJkCOnwhV"


class MIR1KStructure(BaseStructure):
    """Constant settings of MIR-1K dataset."""

    #: Download url of MIR-1K dataset
    url = "http://mirlab.org/dataset/public/MIR-1K.zip"

    #: Label extension
    label_ext = ".pv"

    #: Folder to train wavs
    train_wavs = ["train_data"]

    #: Folder to train labels
    train_labels = ["PitchLabel"]

    #: Folder to test wavs
    test_wavs = ["test_data"]

    #: Folder to test labels
    test_labels = ["PitchLabel"]

    #: Percentage of training data to be partitioned.
    percent_train = 0.88

    @classmethod
    def _post_download(cls, dataset_path):
        """Re-distribute wav files into training and testing data."""
        logger.debug("Received dataset path: %s", dataset_path)
        wavs = _get_file_list(dataset_path, ["Wavfile"], ".wav")
        train_num = round(len(wavs) * cls.percent_train)
        train_wavs = wavs[:train_num]
        test_wavs = wavs[train_num:]
        train_folder = os.path.join(dataset_path, "train_data")
        test_folder = os.path.join(dataset_path, "test_data")
        ensure_path_exists(train_folder)
        ensure_path_exists(test_folder)
        for wav in train_wavs:
            copy(wav, train_folder)
        for wav in test_wavs:
            copy(wav, test_folder)
        return wavs


class CMediaStructure(BaseStructure):
    """Constant settings of CMedia dataset."""

    #: Official download url provided by MIREX.
    url = "https://drive.google.com/uc?export=download&id=15b298vSP9cPP8qARQwa2X_0dbzl6_Eu7"

    #: Label extension
    label_ext = ".csv"

    #: Folder to train wavs
    train_wavs = ["train_data"]

    #: Folder to train labels
    train_labels = ["train_labels"]

    #: Folder to test wavs
    test_wavs = []

    #: Folder to test labels
    test_labels = []


class TonasStructure(BaseStructure):
    """Constant settings of TONAS dataset."""

    #: The dataset is not made public. You have to ask for the access from zenodo:
    #: https://zenodo.org/record/1290722
    url = None

    #: Label extension for note-level transcription.
    label_ext = ".notes.Corrected"

    #: Label extension for f0 contour transcription.
    label_f0_ext = ".f0.Corrected"

    #: Folder to train wavs
    train_wavs = ["Deblas", "Martinetes1", "Martinetes2"]

    #: Folder to train labels
    train_labels = train_wavs

    #: Folder to test wavs
    test_wavs = []

    #: Folder to test labels
    test_labels = []
