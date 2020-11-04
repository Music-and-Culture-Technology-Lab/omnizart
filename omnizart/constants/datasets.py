"""Information about directory structure of each dataset."""
# pylint: disable=C0112
import os
import glob
from abc import abstractmethod, ABCMeta


class BaseStructure(metaclass=ABCMeta):
    """Defines the necessary attributes and methods for each sub-dataset structure class.

    All sub-dataset structure class should inherit this meta class to ensure
    the necessary attributes, methods are overriden.
    """
    @property
    @classmethod
    def url(cls):
        dataset = str(cls.__class__).replace('Structure', '')  # noqa: F841 # pylint: disable=W0612
        raise AttributeError("Attribute 'url' not defined for dataset {dataset}")

    @property
    @abstractmethod
    def label_ext(self):
        """The extension of ground-truth files.

        Examples
        --------
        Defines a sub-dataset structure class:

        class SubStructure(BaseStructure):
            @property
            def label_ext(self):
                return '.mid'
        """

    @property
    @abstractmethod
    def train_wavs(self):
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

    @property
    @abstractmethod
    def test_wavs(self):
        """Records folders that contains testing wav files.

        Same as what `train_wavs` does, but for testing wav files.
        """

    @property
    @abstractmethod
    def train_labels(self):
        """Records folders that contains training labels.

        Similar to the `train_wavs` function, records information of where the corresponding
        ground-truth files are stored.
        """

    @property
    @abstractmethod
    def test_labels(self):
        """Records folders that contains testing labels.

        Similar to the `train_labels` function, records information of where the corresponding
        ground-truth files are stored.
        """

    def _get_file_list(self, dataset_path, dirs, ext):  # pylint: disable=R0201
        files = []
        for _dir in dirs:
            files += glob.glob(os.path.join(dataset_path, _dir, ext))
        return files

    def get_train_wavs(self, dataset_path="./"):
        """Get list of complete train wav paths"""
        return self._get_file_list(dataset_path, self.train_wavs, ".wav")

    def get_test_wavs(self, dataset_path="./"):
        """Get list of complete test wav paths"""
        return self._get_file_list(dataset_path, self.test_wavs, ".wav")

    def get_train_labels(self, dataset_path="./"):
        """Get list of complete train label paths"""
        return self._get_file_list(dataset_path, self.train_labels, self.label_ext)

    def get_test_labels(self, dataset_path="./"):
        """Get list of complete test label paths"""
        return self._get_file_list(dataset_path, self.test_labels, self.label_ext)


class MapsStructure(BaseStructure):
    """Structure of MAPS dataset"""
    @property
    def label_ext(self):
        """"""
        return ".txt"

    @property
    def train_wavs(self):
        """"""
        return [
            "MAPS_AkPnBcht_2/AkPnBcht/MUS",
            "MAPS_AkPnBsdf_2/AkPnBsdf/MUS",
            "MAPS_AkPnStgb_2/AkPnStgb/MUS",
            "MAPS_AkPnCGdD_2/AkPnCGdD/MUS",
            "MAPS_SptkBGCl_2/SptKBGCl/MUS",
            "MAPS_StbgTGd2_2/StbgTGd2/MUS",
        ]

    @property
    def test_wavs(self):
        """"""
        return ["MAPS_ENSTDkAm_2/ENSTDkAm/MUS", "MAPS_ENSTDkCl_2/ENSTDkCl/MUS"]

    @property
    def train_labels(self):
        """"""
        return self.train_wavs

    @property
    def test_labels(self):
        """"""
        return self.test_wavs


class MusicNetStructure(BaseStructure):
    """Structure of MusicNet dataset"""

    #: Dataset URL
    url = "https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz"

    @property
    def label_ext(self):
        """"""
        return ".csv"

    @property
    def train_wavs(self):
        """"""
        return ["train_data"]

    @property
    def test_wavs(self):
        """"""
        return ["test_data"]

    @property
    def train_labels(self):
        """"""
        return ["train_labels"]

    @property
    def test_labels(self):
        """"""
        return ["test_labels"]


class MaestroStructure(BaseStructure):
    """Structure of Maestro dataset"""

    #: Dataset URL
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip"

    @property
    def label_ext(self):
        """"""
        return ".midi"

    @property
    def train_wavs(self):
        """"""
        return ["2004", "2006", "2008", "2009", "2011", "2013", "2014", "2015", "2017"]

    @property
    def test_wavs(self):
        """"""
        return ["2018"]

    @property
    def train_labels(self):
        """"""
        return self.train_wavs

    @property
    def test_labels(self):
        """"""
        return self.test_wavs


class PopStructure(BaseStructure):
    """Structure of Pop dataset"""
    @property
    def label_ext(self):
        """"""
        return ".mid"

    @property
    def train_wavs(self):
        """"""
        return [
            "01_ytd_audio/dist0p00",
            "01_ytd_audio/dist0p10",
            "01_ytd_audio/dist0p20",
            "01_ytd_audio/dist0p40",
            "01_ytd_audio/dist0p50",
            "01_ytd_audio/dist0p60",
        ]

    @property
    def test_wavs(self):
        """"""
        return ["01_ytd_audio/dist0p30"]

    @property
    def train_labels(self):
        """"""
        return [
            "05_align_mid/dist0p00",
            "05_align_mid/dist0p10",
            "05_align_mid/dist0p20",
            "05_align_mid/dist0p40",
            "05_align_mid/dist0p50",
            "05_align_mid/dist0p60",
        ]

    @property
    def test_labels(self):
        """"""
        return ["05_align_mid/dist0p30"]


class ExtSuStructure(BaseStructure):
    """Structure of Extended-Su dataset"""

    #: Dataset URL
    url = "https://drive.google.com/uc?export=download&id=1Miw9G2O1Y8g253RQ2uQ4udM5XMKB-9p8"

    @property
    def label_ext(self):
        """"""
        return ".mid"

    @property
    def train_wavs(self):
        """"""
        return []

    @property
    def test_wavs(self):
        """"""
        return [
            "1 Tchaikovsky",
            "2 schumann",
            "3 beethoven",
            "5 Mozart",
            "PQ01_Dvorak",
            "PQ02_Elgar",
            "PQ03_Farranc",
            "PQ04_Frank",
            "PQ05_Hummel",
            "PQ06_Schostakovich",
            "PQ07_Schubert",
            "PQ08_Schubert",
            "SQ01_Beethoven",
            "SQ02_Janacek",
            "SQ03_Schubert",
            "SQ04_Janacek",
            "SQ04_Ravel",
            "SQ05_Mozart",
            "SQ07_Haydn",
            "SQ08_Dvorak",
            "SQ09_Ravel",
            "SY06_Mahler",
            "VS01_Schumann",
            "VS02_Brahms",
            "VS03_Debussy",
            "VS04_Franck",
            "VS05_Mozart",
            "WQ01_Nielsen",
            "WQ02_Schoenberg",
            "WQ03_Cambini",
            "WQ04_Danzi",
        ]

    @property
    def train_labels(self):
        """"""
        return self.train_wavs

    @property
    def test_labels(self):
        """"""
        return self.test_wavs


class McGillBillBoard:  # pylint: disable=R0903
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


class BeethovenSonatas:  # pylint: disable=R0903
    """Constant settings of BPS-FH dataset"""

    #: Hosted dataset download url.
    url = "https://drive.google.com/uc?export=download&id=1nYq2FB5LQfYJoXyYZl3XcklpJkCOnwhV"
