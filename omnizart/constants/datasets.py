"""Information about each dataset.

Supports an easy way to deal with complex folder structure of different
datasets in a consice interface.
Partial support for downloading public datasets.

Complete information of supported datasets are as following:

+-----------+--------------+-----------+--------------+
| Dataset   | Downloadable | Dataset   | Downloadable |
+===========+==============+===========+==============+
| Maps      |              | BPS-FH    | O            |
+-----------+--------------+-----------+--------------+
| MusicNet  | O            | MIR-1K    | O            |
+-----------+--------------+-----------+--------------+
| Maestro   | O            | CMedia    |              |
+-----------+--------------+-----------+--------------+
| Pop       |              | Tonas     |              |
+-----------+--------------+-----------+--------------+
| Ext-Su    | O            | MedleyDB  |              |
+-----------+--------------+-----------+--------------+
| BillBoard | O            |           |              |
+-----------+--------------+-----------+--------------+

"""
# pylint: disable=C0112
import os
import csv
import glob
from os.path import join as jpath
from shutil import copy

import pretty_midi
import numpy as np

from omnizart.io import load_yaml
from omnizart.base import Label
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
    """Record folders that contain trainig wav files.

    The path to sub-folders should be the relative path to root folder of the dataset.

    Examples
    --------
    Assume the structure of the dataset looks like following

    |  maps
    |  ├── MAPS_AkPnCGdD_2
    |  │   └── AkPnCGdD
    |  └── MAPS_AkPnBcht_2
    |      └── AkPnBcht

    where ``AkPnCGdD`` and ``AkPnBcht`` are the folders that store the wav files.
    The function should then return a list like:

    |  >>> ['MAPS_AkPnCGdD_2/AkPnCGdD', 'MAPS_AkPnBcht_2/AkPnBcht']
    """

    test_wavs = None
    """Record folders that contain testing wav files.

    Same as what `train_wavs` does, but for testing wav files.
    """

    train_labels = None
    """Record folders that contains training labels.

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
        # Transform the basename of wav file to the corressponding label file name.
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
        dataset_path, unzip_done = download_large_file_from_google_drive(
            cls.url, save_path=save_path, save_name=save_name, unzip=True
        )
        if unzip_done:
            os.remove(save_name)
        cls._post_download(dataset_path=dataset_path)

    @classmethod
    def _post_download(cls, dataset_path):
        pass

    @classmethod
    def load_label(cls, label_path):
        """Load and parse labels for the given label file path.

        Parses different format of label information to shared intermediate format,
        encapslated with :class:`Label` instances. The default is parsing MIDI
        file format.
        """
        midi = pretty_midi.PrettyMIDI(label_path)
        labels = []
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                label = Label(
                    start_time=note.start,
                    end_time=note.end,
                    note=note.pitch,
                    velocity=note.velocity,
                    instrument=inst.program
                )
                if label.note == -1:
                    continue
                labels.append(label)
        return labels


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

    @classmethod
    def load_label(cls, label_path):
        lines = open(label_path, "r").readlines()[1:]  # Discard the first line which contains column names
        labels = []
        for line in lines:
            if line.strip() == "":
                continue
            values = line.split("\t")
            onset, offset, note = float(values[0]), float(values[1]), int(values[2].strip())
            labels.append(Label(start_time=onset, end_time=offset, note=note))
        return labels


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

    @classmethod
    def load_label(cls, label_path):
        labels = []
        sample_rate = 44100
        with open(label_path, "r") as label_file:
            reader = csv.DictReader(label_file, delimiter=",")
            for row in reader:
                onset = float(row["start_time"]) / sample_rate
                offset = float(row["end_time"]) / sample_rate
                inst = int(row["instrument"]) - 1
                note = int(row["note"])

                # The statement used in the paper is 'measure', which is kind of ambiguous.
                start_beat = float(row["start_beat"])

                # It's actually beat length of 'end_beat' column, thus adding start beat position here to
                # make it a 'real end_beat'.
                end_beat = float(row["end_beat"]) + start_beat
                note_value = row["note_value"]

                label = Label(
                    start_time=onset,
                    end_time=offset,
                    note=note,
                    instrument=inst,
                    start_beat=start_beat,
                    end_beat=end_beat,
                    note_value=note_value
                )
                labels.append(label)
        return labels


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

    #: Split ID of train/test set.
    train_test_split_id = 1000

    #: Dataset URL
    url = "https://drive.google.com/uc?export=download&id=1_K_Fof4zt1IQvs1aDmf-5wY0wHqgcPlC"

    #: File IDs to be ignored
    ignore_ids = [353, 634, 1106]

    @classmethod
    def _get_train_test_split_ids(cls, dataset_path):
        """Get train/test set split indexes.

        Default will use the folder ID as the partition base.
        The index number smaller than `train_test_split_id` will be taken as the
        training set, and others for testing set.

        Returns
        -------
        train_ids: list[str]
            Folder ids of training set.
        test_ids: list[str]
            Folder ids of testing set
        """
        index_file_path = jpath(dataset_path, cls.index_file_path)
        reader = csv.DictReader(open(index_file_path, "r"), delimiter=",")
        name_id_mapping = {}
        for data in reader:
            pid = int(data["id"])
            if data["title"] != "" and pid not in cls.ignore_ids:
                name = data["artist"] + ": " + data["title"]
                if name not in name_id_mapping:
                    name_id_mapping[name] = []
                name_id_mapping[name].append(pid)  # Repetition count: 1->613, 2->110, 3->19

        train_ids, test_ids = [], []  # Folder ids
        for pids in name_id_mapping.values():
            if len(pids) <= 2:
                pid = pids[0]
            else:
                pid = pids[2]

            if pid <= cls.train_test_split_id:
                train_ids.append(str(pid).zfill(4))
            else:
                test_ids.append(str(pid).zfill(4))

        return train_ids, test_ids

    @classmethod
    def _get_paths_in_ids(cls, target_folder, target_file, ids):
        output = []
        for f_name in os.listdir(target_folder):
            if f_name in ids:
                output.append(jpath(target_folder, f_name, target_file))
        return output

    @classmethod
    def get_train_wavs(cls, dataset_path):
        train_ids, _ = cls._get_train_test_split_ids(dataset_path)
        feat_path = jpath(dataset_path, cls.feature_folder)
        return cls._get_paths_in_ids(feat_path, cls.feature_file_name, train_ids)

    @classmethod
    def get_train_labels(cls, dataset_path):
        train_ids, _ = cls._get_train_test_split_ids(dataset_path)
        label_path = jpath(dataset_path, cls.label_folder)
        return cls._get_paths_in_ids(label_path, cls.label_file_name, train_ids)

    @classmethod
    def get_test_wavs(cls, dataset_path):
        _, test_ids = cls._get_train_test_split_ids(dataset_path)
        feat_path = jpath(dataset_path, cls.feature_folder)
        return cls._get_paths_in_ids(feat_path, cls.feature_file_name, test_ids)

    @classmethod
    def get_test_labels(cls, dataset_path):
        _, test_ids = cls._get_train_test_split_ids(dataset_path)
        label_path = jpath(dataset_path, cls.label_folder)
        return cls._get_paths_in_ids(label_path, cls.label_file_name, test_ids)

    @classmethod
    def _get_data_pair(cls, wavs, labels):
        wavs = [os.path.abspath(wav) for wav in wavs]
        labels = [os.path.abspath(label) for label in labels]
        get_id = lambda path: os.path.basename(os.path.dirname(path))
        label_path_mapping = {get_id(label): label for label in labels}

        pair = []
        for wav in wavs:
            wav_id = get_id(wav)
            label_path = label_path_mapping[wav_id]
            assert os.path.exists(wav)
            assert os.path.exists(label_path)
            pair.append((wav, label_path))
        return pair


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

    @classmethod
    def load_label(cls, label_path):
        with open(label_path, "r") as lin:
            lines = lin.readlines()

        notes = np.array([round(float(note)) for note in lines])
        note_diff = notes[1:] - notes[:-1]
        change_idx = np.where(note_diff != 0)[0] + 1
        change_idx = np.insert(change_idx, 0, 0)  # Padding a single zero to the beginning.
        labels = []
        for idx, chi in enumerate(change_idx[:-1]):
            note = notes[chi]
            if note == 0:
                continue

            start_t = 0.01 * chi + 0.02  # The first frame starts from 20ms.
            end_t = 0.01 * change_idx[idx+1] + 0.02  # noqa: E226
            if end_t - start_t < 0.05:
                # Minimum duration should over 50ms.
                continue

            labels.append(Label(
                start_time=float(start_t),
                end_time=float(end_t),
                note=note
            ))
        return labels


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

    @classmethod
    def load_label(cls, label_path):
        labels = []
        with open(label_path, "r") as label_file:
            reader = csv.DictReader(label_file, delimiter=",")
            for row in reader:
                labels.append(Label(
                    start_time=float(row["onset"]),
                    end_time=float(row["offset"]),
                    note=int(row["note"])
                ))
        return labels


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

    @classmethod
    def load_label(cls, label_path):
        with open(label_path, "r") as lin:
            lines = lin.readlines()

        labels = []
        for line in lines[1:]:
            onset, dura, note, _ = line.split(", ")
            labels.append(Label(
                start_time=float(onset),
                end_time=float(onset) + float(dura),
                note=round(float(note))
            ))
        return labels


class MedleyDBStructure(BaseStructure):
    """Constant settings of MedleyDB dataset.

    Notice here that current design of the structure is for getting
    vocal melody ground-truth files only. All other tracks are ignored
    when calling to ``get_train_data_pair`` or ``get_test_data_pair``.
    """
    #: Folder to train wavs
    train_wavs = [
        'Audio/AClassicEducation_NightOwl', 'Audio/AimeeNorwich_Child', 'Audio/AimeeNorwich_Flying',
        'Audio/AlexanderRoss_GoodbyeBolero', 'Audio/AlexanderRoss_VelvetCurtain', 'Audio/AmarLal_Rest',
        'Audio/AmarLal_SpringDay1', 'Audio/Auctioneer_OurFutureFaces', 'Audio/AvaLuna_Waterduct',
        'Audio/BigTroubles_Phantom', 'Audio/BrandonWebster_DontHearAThing', 'Audio/BrandonWebster_YesSirICanFly',
        'Audio/CelestialShore_DieForUs', 'Audio/ChrisJacoby_BoothShotLincoln',
        'Audio/ClaraBerryAndWooldog_Boys', 'Audio/ClaraBerryAndWooldog_Stella',
        'Audio/ClaraBerryAndWooldog_TheBadGuys', 'Audio/ClaraBerryAndWooldog_WaltzForMyVictims',
        'Audio/Creepoid_OldTree', 'Audio/CroqueMadame_Oil', 'Audio/CroqueMadame_Pilot',
        'Audio/Debussy_LenfantProdigue', 'Audio/DreamersOfTheGhetto_HeavyLove',
        'Audio/EthanHein_1930sSynthAndUprightBass', 'Audio/EthanHein_BluesForNofi', 'Audio/EthanHein_GirlOnABridge',
        'Audio/EthanHein_HarmonicaFigure', 'Audio/FamilyBand_Again', 'Audio/Grants_PunchDrunk',
        'Audio/Handel_TornamiAVagheggiar', 'Audio/HeladoNegro_MitadDelMundo', 'Audio/HezekiahJones_BorrowedHeart',
        'Audio/HopAlong_SisterCities', 'Audio/InvisibleFamiliars_DisturbingWildlife',
        'Audio/JoelHelander_ExcessiveResistancetoChange', 'Audio/JoelHelander_IntheAtticBedroom',
        'Audio/KarimDouaidy_Hopscotch', 'Audio/KarimDouaidy_Yatora', 'Audio/LizNelson_Coldwar',
        'Audio/LizNelson_ImComingHome', 'Audio/LizNelson_Rainfall', 'Audio/MatthewEntwistle_AnEveningWithOliver',
        'Audio/MatthewEntwistle_DontYouEver', 'Audio/MatthewEntwistle_FairerHopes',
        'Audio/MatthewEntwistle_ImpressionsOfSaturn', 'Audio/MatthewEntwistle_Lontano',
        'Audio/MatthewEntwistle_TheFlaxenField', 'Audio/Meaxic_TakeAStep', 'Audio/Meaxic_YouListen',
        'Audio/MichaelKropf_AllGoodThings', 'Audio/Mozart_BesterJungling', 'Audio/Mozart_DiesBildnis',
        'Audio/MusicDelta_80sRock', 'Audio/MusicDelta_Beatles', 'Audio/MusicDelta_BebopJazz',
        'Audio/ChrisJacoby_PigsFoot', 'Audio/FacesOnFilm_WaitingForGa', 'Audio/Lushlife_ToynbeeSuite',
        'Audio/MusicDelta_Beethoven', 'Audio/MusicDelta_Grunge', 'Audio/Phoenix_BrokenPledgeChicagoReel',
        'Audio/MusicDelta_Britpop', 'Audio/MusicDelta_ChineseChaoZhou', 'Audio/MusicDelta_ChineseDrama',
        'Audio/MusicDelta_ChineseHenan', 'Audio/MusicDelta_ChineseJiangNan', 'Audio/MusicDelta_ChineseXinJing',
        'Audio/MusicDelta_ChineseYaoZu', 'Audio/MusicDelta_CoolJazz', 'Audio/MusicDelta_Country1',
        'Audio/MusicDelta_Country2', 'Audio/MusicDelta_Disco', 'Audio/MusicDelta_FreeJazz',
        'Audio/MusicDelta_FusionJazz', 'Audio/MusicDelta_Gospel', 'Audio/MusicDelta_GriegTrolltog',
        'Audio/MusicDelta_Hendrix', 'Audio/MusicDelta_InTheHalloftheMountainKing',
        'Audio/MusicDelta_LatinJazz', 'Audio/MusicDelta_ModalJazz', 'Audio/MusicDelta_Pachelbel',
        'Audio/MusicDelta_Punk', 'Audio/MusicDelta_Reggae', 'Audio/MusicDelta_Rock',
        'Audio/MusicDelta_Shadows', 'Audio/MusicDelta_SpeedMetal', 'Audio/MusicDelta_SwingJazz',
        'Audio/MusicDelta_Vivaldi', 'Audio/MusicDelta_Zeppelin', 'Audio/NightPanther_Fire',
        'Audio/Phoenix_ColliersDaughter', 'Audio/Phoenix_ElzicsFarewell',
        'Audio/Phoenix_ScotchMorris', 'Audio/Phoenix_SeanCaughlinsTheScartaglen', 'Audio/PortStWillow_StayEven',
        'Audio/PurlingHiss_Lolita', 'Audio/Schubert_Erstarrung', 'Audio/Schumann_Mignon',
        'Audio/StrandOfOaks_Spacestation', 'Audio/SweetLights_YouLetMeDown',
        'Audio/TablaBreakbeatScience_CaptainSky', 'Audio/TablaBreakbeatScience_MiloVsMongo',
        'Audio/TablaBreakbeatScience_MoodyPlucks', 'Audio/TablaBreakbeatScience_PhaseTransition',
        'Audio/TablaBreakbeatScience_RockSteady', 'Audio/TablaBreakbeatScience_Scorpio',
        'Audio/TablaBreakbeatScience_Vger', 'Audio/TablaBreakbeatScience_WhoIsIt', 'Audio/TheDistricts_Vermont',
        'Audio/TheScarletBrand_LesFleursDuMal', 'Audio/TheSoSoGlos_Emergency', 'Audio/Wolf_DieBekherte'
    ]

    #: Folder to test wavs
    test_wavs = [
        'Audio/MusicDelta_FunkJazz',
        'Audio/Phoenix_LarkOnTheStrandDrummondCastle',
        'Audio/MatthewEntwistle_TheArch',
        'Audio/SecretMountains_HighHorse',
        'Audio/Snowmine_Curfews',
        'Audio/StevenClark_Bounty',
        'Audio/TablaBreakbeatScience_Animoog',
        'Audio/ClaraBerryAndWooldog_AirTraffic',
        'Audio/MusicDelta_Rockabilly',
        'Audio/JoelHelander_Definition'
    ]

    #: Folder to train label files
    train_labels = ["Annotations/Pitch_Annotations"]

    #: Folder to test label files
    test_labels = ["Annotations/Pitch_Annotations"]

    #: Folder to pitch-related ground truth files.
    #: The reason to define another variable for holding the same content
    #: as ``train_labels`` is to keep the flexibility for extending the
    #: ability getting other kinds of ground-truth files such as guitar
    #: tracks or bass tracks.
    pitch_annotation_folder = "Annotations/Pitch_Annotations"

    #: Extension of pitch label files.
    label_ext = ".csv"

    #: Postfix of meta files.
    meta_file_postfix = "_METADATA.yaml"

    #: Postfix of wav files.
    wav_postfix = "_MIX"

    #: Postfix of pitch label files. Can be formatted with
    #: ``label_postfix.format(track_num=...)``.
    label_postfix = "_STEM_{track_num}"

    # Override functions to fit the scenario of getting vocal ground-truth
    # files only.
    @classmethod
    def _get_label_files(cls, dataset_path, wav_paths):
        # Get only vocal ground-truth files
        labels = []
        for wav_path in wav_paths:
            filename, _ = os.path.splitext(os.path.basename(wav_path))
            filename = filename.replace(cls.wav_postfix, "")
            meta_file = filename + cls.meta_file_postfix
            meta_file = os.path.join(os.path.dirname(wav_path), meta_file)
            meta = load_yaml(meta_file)

            if meta["instrumental"] == "yes":
                # Ignore instrumental songs that have no vocals.
                continue

            for tid, track in meta["stems"].items():
                if "singer" not in track["instrument"]:
                    # Ignore instruments
                    continue

                label_name = filename + cls.label_postfix.format(track_num=tid[1:]) + cls.label_ext
                label = os.path.join(dataset_path, cls.pitch_annotation_folder, label_name)
                if not os.path.exists(label):
                    # Not the main melody vocal
                    pass
                else:
                    labels.append(label)
                    break
        return labels

    @classmethod
    def get_train_labels(cls, dataset_path):
        return cls._get_label_files(dataset_path, cls.get_train_wavs(dataset_path))

    @classmethod
    def get_test_labels(cls, dataset_path):
        return cls._get_label_files(dataset_path, cls.get_test_wavs(dataset_path))

    @classmethod
    def _get_data_pair(cls, wavs, labels):
        label_path_mapping = {}
        for label in labels:
            basename = os.path.basename(label)
            true_name = basename.split("_STEM_")[0]
            label_path_mapping[true_name] = label

        pair = []
        for wav in wavs:
            filename, _ = os.path.splitext(os.path.basename(wav))
            label_name = cls._name_transform(filename)
            if label_name not in label_path_mapping:
                # The song has no vocal track, thus skip.
                continue
            label_path = label_path_mapping[label_name]
            pair.append((wav, label_path))
        return pair

    @classmethod
    def _name_transform(cls, basename):
        return basename.replace(cls.wav_postfix, "")

    @classmethod
    def load_label(cls, label_path):
        with open(label_path, "r") as fin:
            lines = fin.readlines()

        labels = []
        t_unit = 256 / 44100  # ~= 0.0058 secs
        for line in lines:
            elems = line.strip().split(",")
            sec, hz = float(elems[0]), float(elems[1])  # pylint: disable=invalid-name
            if hz < 1e-10:
                continue
            note = float(pretty_midi.hz_to_note_number(hz))  # Convert return type of np.float64 to float
            end_t = sec + t_unit
            labels.append(Label(start_time=sec, end_time=end_t, note=note))

        return labels
