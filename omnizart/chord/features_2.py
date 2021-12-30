# pylint: disable=W0102,C0103
import math

import numpy as np
import joblib
import numpy.lib.recfunctions as rfn
from scipy.ndimage import gaussian_filter1d
import os
from os.path import join as jpath

# from omnizart.constants.feature import CHORD_INT_MAPPING, ENHARMONIC_TABLE
from omnizart.constants.feature import CHORD_INT_MAPPING_2


def extract_feature_label(feat_path, lab_path, audio_sr=22050, hop_size=1024):
    """Basic feature extraction block.

    Parameters
    ----------
    feat_path: Path
        Path to the raw feature folder.
    lab_path: Path
        Path to the corresponding label folder.
    audio_sr: int
        sampling rate, default=22050.
    hop_size: int
        number of samples between successive CQT columns, default=1024.

    Returns
    -------
    data:
        Processed data
    """
    label = load_label(lab_path)
    feature = load_feature(feat_path)

    pitch_shift = feat_path.replace(".pickle", "").split(':pitch_shift=')[1]
    pitch_shift = int(pitch_shift)
    cqt = feature['cqt']
    n_frames = cqt.shape[0]

    # Get frame-wise labels
    chords = np.zeros(n_frames, dtype=np.int32)
    for lab in label:
        onset_idx = int(lab['onset'] * audio_sr / hop_size)
        end_idx = int(math.ceil(lab['end'] * audio_sr / hop_size))
        chord = CHORD_INT_MAPPING_2[lab['chord']]
        chords[onset_idx:end_idx] = chord

    # Chord labels modulation
    chords_shift = shift_chord_labels(chords, pitch_shift)

    # Chord transition
    transition = get_chord_transition(chords_shift)

    data = {'cqt': cqt, 'chord': chords_shift, 'transition': transition}

    # Reshape
    data = reshape_data(data)

    return data


def get_label_dict(dataset_path, label_folder="annotations/chordlab/The Beatles"):
    """Get dictionary with keys = track ids and values = label paths."""
    label_dirs = [os.path.normpath(jpath(subdir, file)) for subdir, dirs, files in
                  os.walk(jpath(dataset_path, label_folder)) for file in files if file.endswith(cls.label_ext)]

    id_fold_mapping = _get_id_fold_mapping(dataset_path)
    valid_ids = id_fold_mapping.keys()
    label_dict = {}
    for _dir in label_dirs:
        id = _label_dir2id(_dir)
        if id not in valid_ids:
            print('Invalid id:', id)
            exit(1)
        label_dict[id] = _dir
    return label_dict


def _get_id_fold_mapping():
    split_path = "https://github.com/superbock/ISMIR2020/blob/master/splits/beatles_8-fold_cv_album_distributed.folds"
    with open(split_path) as sfile:
        lines = [(line.split('\t')[0], int(line.split('\t')[1].replace(r'\n', ''))) for line in sfile.readlines()]
        id_fold_mapping = {line[0]: line[1] for line in lines}
    return id_fold_mapping


def _label_dir2id(_dir):
    """Get label id from label dir"""
    sdir = os.path.normpath(_dir).split(os.path.sep)
    track_id = sdir[-2].replace('_-_', '_').replace('\'', '').replace('!', '').replace('.', '')
    album_id = sdir[-1].replace('CD1_-_', '').replace('CD2_-_', '').replace('.lab', '').replace('_-_', '_')
    album_id = album_id.replace('\'', '').replace(',', '').replace('!', '').replace('.', '')
    id = 'beatles_' + album_id + '_' + track_id
    return id


def load_label(lab_path):
    """Load and parse the label into the desired format for later process."""
    labels = np.genfromtxt(lab_path, dtype=[('onset', np.float32), ('end', np.float32), ('chord', '<U24')])
    chords = np.array([_format_chord(c) for c in labels['chord']], dtype=[('root', '<U3'), ('attribute', '<U24')])
    chords_renamed = np.array([_rename_chord(c) for c in chords], dtype=[('chord', '<U10')])
    new_labels = rfn.merge_arrays([labels[['onset', 'end']], chords_renamed], flatten=True, usemask=False)
    return new_labels


def _format_chord(chord):
    """Get root and attribute of chord."""
    split_idx = 1 if len(chord) == 1 or chord[1] != 'b' else 2
    root = chord[:split_idx]
    attribute = chord[split_idx:]
    return (root, attribute)


def _rename_chord(chord):
    """Rename chord. Default: 26 chord classes, 12 maj + 12 min + other + non-chord."""
    root, attribute = chord['root'], chord['attribute']
    attribute = attribute.split('/')[0]  # remove inversion
    if root == 'N':  # non-chord
        return root
    elif any(s in attribute for s in [':min', ':minmaj']):  # minor
        return root + ':min'
    elif attribute == '' or any(s in attribute for s in [':maj', ':7', ':9']):  # major
        return root + ':maj'
    elif any(s in attribute for s in [':(', ':aug', ':dim', ':hdim7', ':sus2', ':sus4']):  # others
        return 'others'
    else:
        print('invalid syntax:', chord, root, attribute)
        exit(1)


def load_features(feat_path):
    """Get CQT features."""
    with open(feat_path, "rb") as f:
        feature = joblib.load(f)
    return feature


def get_data_pair(cls, dataset_path, audio_sr=22050, hop_size=1024):
    feature_dict = cls.get_features(dataset_path)
    label_dict = cls.get_labels(dataset_path)

    data_dict = {}
    for feature_id in list(feature_dict.keys()):
        label_id, pitch_shift = feature_id.split(':pitch_shift=')
        pitch_shift = int(pitch_shift)
        cqt = feature_dict[feature_id]['cqt']
        labels = label_dict[label_id]
        n_frames = cqt.shape[0]

        # Get frame-wise labels
        chords = np.zeros(n_frames, dtype=np.int32)
        for label in labels:
            onset_idx = int(label['onset'] * audio_sr / hop_size)
            end_idx = int(math.ceil(label['end'] * audio_sr / hop_size))
            chord = CHORD_INT_MAPPING_2[label['chord']]
            chords[onset_idx:end_idx] = chord

        # Chord labels modulation
        chords_shift = cls.shift_chord_labels(chords, pitch_shift)

        # Chord transition
        transition = cls.get_chord_transition(chords_shift)

        data_dict[feature_id] = {'cqt': cqt, 'chord': chords_shift, 'transition': transition}
    return data_dict


def shift_chord_labels(chords, shift):
    chords_shift = np.where(chords < 12, (chords + shift) % 12, chords)
    chords_shift = np.where((chords >= 12) & (chords < 24), (chords_shift - 12 + shift) % 12 + 12,
                            chords_shift)
    return chords_shift


def get_chord_transition(chords):
    return np.array([1] + [0 if f2 == f1 else 1 for f1, f2 in zip(chords[:-1], chords[1:])], dtype=np.int32)


def log_compression(x, gamma=1):
    return np.log(1 + gamma*x)


def reshap_data(data, seq_len=256, seq_hop=128, min_seq_len=128):
    cqt = data['cqt']
    n_frames = cqt.shape[0]
    cqt = log_compression(cqt, gamma=100)
    y_c = data['chord']
    y_t = data['transition']
    frame_ids = [k + ':' + str(i) for i in range(n_frames)]

    # Segment
    cqt_reshape = [cqt[i:i+seq_len] for i in range(0, n_frames, seq_hop) if n_frames - i >= min_seq_len]
    y_c_reshape = [y_c[i:i+seq_len] for i in range(0, n_frames, seq_hop) if n_frames - i >= min_seq_len]
    y_t_reshape = [y_t[i:i+seq_len] for i in range(0, n_frames, seq_hop) if n_frames - i >= min_seq_len]
    valid_lens = np.array([len(x) for x in cqt_reshape])
    frame_ids_reshape = [frame_ids[i:i+seq_len] for i in range(0, n_frames, seq_hop) if n_frames - i >= min_seq_len]

    # Pad and Stack
    cqt_reshape = np.stack([np.pad(x, pad_width=[(0, seq_len - len(x)), (0, 0)], mode='constant') for x in cqt_reshape])
    y_c_reshape = np.stack(
        [np.pad(x, pad_width=[(0, seq_len - len(x))], mode='constant', constant_values=25) for x in y_c_reshape]
    )
    y_t_reshape = np.stack(
        [np.pad(x, pad_width=[(0, seq_len - len(x))], mode='constant', constant_values=0) for x in y_t_reshape]
    )
    pad_id = k + ':pad'
    frame_ids_reshape = np.stack(
        [np.pad(x, pad_width=[(0, seq_len - len(x))], mode='constant', constant_values=pad_id) for x in
         frame_ids_reshape]
    )

    data_reshape = {'cqt': cqt_reshape,
                    'chord': y_c_reshape,
                    'trainsition': y_t_reshape,
                    'len': valid_lens,
                    'frame_id': frame_ids_reshape}

    return data_reshape
