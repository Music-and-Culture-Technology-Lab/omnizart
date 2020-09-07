"""Information about directory structure of each dataset."""
# pylint: disable=C0112
from abc import abstractmethod, ABCMeta


class BaseStructure(metaclass=ABCMeta):
    """Defines the necessary attributes and methods for each sub-dataset structure class.

    All sub-dataset structure class should inherit this meta class to ensure
    the necessary attributes, methods are overriden.
    """
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
        return ["tets_labels"]


class MaestroStructure(BaseStructure):
    """Structure of Maestro dataset"""
    @property
    def label_ext(self):
        """"""
        return ".midi"

    @property
    def train_wavs(self):
        """"""
        return ["2004", "2006", "2008", "2009", "2011", "2013", "2014", "2015"]

    @property
    def test_wavs(self):
        """"""
        return ["2017"]

    @property
    def train_labels(self):
        """"""
        return self.train_wavs

    @property
    def test_labels(self):
        """"""
        return self.test_wavs


class RhythmStructure(BaseStructure):
    """Structure of Rhythm dataset"""
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
