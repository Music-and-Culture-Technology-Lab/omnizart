# pylint: disable=W0102,C0103
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from omnizart.constants.feature import CHORD_INT_MAPPING, ENHARMONIC_TABLE


def extract_feature_label(feat_path, lab_path, segment_width=21, segment_hop=5, num_steps=100):
    """Basic feature extraction block.

    Including multiple steps for processing the feature.
    Steps include:

    * Feature augmentation
    * Feature segmentation
    * Feature reshaping

    Parameters
    ----------
    feat_path: Path
        Path to the raw feature folder.
    lab_path: Path
        Path to the corresponding label folder.
    segment_width: int
        Width of each frame after segementation.
    segment_hop: int
        Hop size for processing each segment.
    num_steps: int
        Number of steps while reshaping the feature.

    Returns
    -------
    feature:
        Processed feature
    """
    label = load_label(lab_path)
    feature = load_feature(feat_path, label)
    feature = augment_feature(feature)
    feature = segment_feature(feature, segment_width=segment_width, segment_hop=segment_hop)
    feature = reshape_feature(feature, num_steps=num_steps)

    return feature


def load_label(lab_path):
    """Load and parse the label into the desired format for later process."""
    label = []
    with open(lab_path, "r") as lab_fp:
        for line in lab_fp:
            data = line.strip().split("\t")
            if len(data) == 3:
                label.append((float(data[0]), float(data[1]), data[2]))

    data_types = [("onset", np.float32), ("end", np.float32), ("chord", object)]
    return np.array(label, dtype=data_types)


def load_feature(feat_path, label):
    """Load and parse the feature into the desired format."""
    frames = {"onset": [], "chroma": [], "chord": [], "chord_change": []}
    pre_chord = None
    for row in np.genfromtxt(feat_path, delimiter=","):
        onset = row[1]
        chroma1 = row[2:14]
        chroma2 = row[14:26]
        both_chroma = np.concatenate([chroma1, chroma2]).astype(np.float32)

        filtered_label = label[(label["onset"] <= onset) & (label["end"] > onset)]
        chord = filtered_label["chord"][0]
        root = chord.split(":")[0]
        if "b" in root:
            root, quality = chord.split(':')
            chord = ENHARMONIC_TABLE[root] + ':' + quality
        chord_int = CHORD_INT_MAPPING[chord]
        chord_change = 0 if chord_int == pre_chord else 1
        pre_chord = chord_int

        frames["onset"].append(onset)
        frames["chroma"].append(both_chroma)
        frames["chord"].append(chord_int)
        frames["chord_change"].append(chord_change)
    return frames


def augment_feature(feature):
    """Feature augmentation

    Variying pitches with 12 different shifts.
    """
    new_feature = []
    for shift in range(12):
        chromagram = np.array(feature["chroma"])
        chord = feature["chord"]
        chord_change = feature["chord_change"]

        chromagram_shift = shift_chromagram(chromagram, shift)
        tc_shift = compute_tonal_centroids((chromagram_shift[:, :12] + chromagram_shift[:, 12:]) / 2)  # [time, 6]
        chord_shift = np.array([shift_chord(x, shift) for x in chord])

        new_feature.append({
            "chroma": chromagram_shift,
            "tc": tc_shift,
            "chord": chord_shift,
            "chord_change": chord_change
        })
    return new_feature


def shift_chromagram(chromagram, shift):
    """Shift chord's chromagram."""
    if shift > 0:
        chr1 = np.roll(chromagram[:, :12], shift, axis=1)
        chr2 = np.roll(chromagram[:, 12:], shift, axis=1)
        chromagram = np.concatenate([chr1, chr2], axis=1)
    return chromagram


def shift_chord(chord, shift):
    """Shift chord"""
    if chord < 12:
        new_chord = (chord + shift) % 12
    elif chord < 24:
        new_chord = (chord - 12 + shift) % 12 + 12
    else:
        new_chord = chord
    return new_chord


def compute_tonal_centroids(chromagram, filtering=True, sigma=8):
    """chromagram with shape [time, 12] """

    # define transformation matrix - phi
    r1, r2, r3 = 1, 1, 0.5
    base_arr = np.arange(12)
    phi_0 = r1 * np.sin(base_arr * 7 * math.pi / 6)
    phi_1 = r1 * np.cos(base_arr * 7 * math.pi / 6)
    phi_2 = r2 * np.sin(base_arr * 3 * math.pi / 2)
    phi_3 = r2 * np.cos(base_arr * 3 * math.pi / 2)
    phi_4 = r3 * np.sin(base_arr * 2 * math.pi / 3)
    phi_5 = r3 * np.cos(base_arr * 2 * math.pi / 3)
    phi_ = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5]
    phi = np.concatenate(phi_).reshape(6, 12)  # [6, 12]
    phi_t = np.transpose(phi)  # [12, 6]

    tc = chromagram.dot(phi_t)  # convert to tonal centroid representations, [time, 6]
    if filtering:
        # Gaussian filtering along time axis
        tc = gaussian_filter1d(tc, sigma=sigma, axis=0)

    return tc.astype(np.float32)


def segment_feature(feature, segment_width=21, segment_hop=5):
    """Partition feature into segments."""
    new_feature = []
    for feat in feature:
        chroma = feat["chroma"]
        tc = feat["tc"]
        chord = feat["chord"]
        chroma_tc = np.concatenate([chroma, tc], axis=1)  # [time, 30]

        pad_size = segment_width // 2
        chroma_tc_pad = np.pad(chroma_tc, ((pad_size, pad_size), (0, 0)), "constant", constant_values=0.0)
        chord_pad = np.pad(chord, (pad_size, pad_size), "constant", constant_values=24)

        num_frames = len(chroma_tc)
        chroma_tc_segment = np.array([
            chroma_tc_pad[i-pad_size:i+pad_size+1]  # noqa: E226
            for i in range(pad_size, pad_size+num_frames, segment_hop)  # noqa: E226
        ])
        chroma_segment = np.reshape(chroma_tc_segment[:, :, :24], [-1, segment_width*24])  # noqa: E226
        tc_segment = np.reshape(chroma_tc_segment[:, :, 24:], [-1, segment_width * 6])
        chord_segment = np.array([chord_pad[i] for i in range(pad_size, pad_size + num_frames, segment_hop)])
        chord_change_segment = np.array(
            [1] + [0 if x == y else 1 for x, y in zip(chord_segment[1:], chord_segment[:-1])]
        )

        new_feature.append({
            "chroma": chroma_segment.astype(np.float32),
            "tc": tc_segment.astype(np.float32),
            "chord": chord_segment.astype(np.int32),
            "chord_change": chord_change_segment.astype(np.int32)
        })
    return new_feature


def reshape_feature(feature, num_steps=100):
    """Reshape the feature into the final output."""
    new_feature = []
    for feat in feature:
        chroma = feat["chroma"]
        tc = feat["tc"]
        chord = feat["chord"]
        chord_change = feat["chord_change"]

        num_frames = len(chroma)
        pad_size = 0 if num_frames/num_steps == 0 else num_steps - num_frames%num_steps  # noqa: E226,E228
        if pad_size != 0:
            chroma = np.pad(chroma, ((0, pad_size), (0, 0)), "constant", constant_values=0)
            tc = np.pad(tc, ((0, pad_size), (0, 0)), "constant", constant_values=0)
            chord = np.pad(chord, (0, pad_size), "constant", constant_values=24)
            chord_change = np.pad(chord_change, (0, pad_size), "constant", constant_values=0)

        seq_hop = num_steps // 2
        num_seqs = int((len(chroma) - num_steps) / seq_hop) + 1
        feat_size = chroma.shape[1]
        tc_size = tc.shape[1]
        s0, s1 = chroma.strides
        chroma_reshape = np.lib.stride_tricks.as_strided(
            chroma, shape=(num_seqs, num_steps, feat_size), strides=(s0 * seq_hop, s0, s1)
        )
        ss0, ss1 = tc.strides
        tc_reshape = np.lib.stride_tricks.as_strided(
            tc, shape=(num_seqs, num_steps, tc_size), strides=(ss0 * seq_hop, ss0, ss1)
        )
        sss0 = chord.strides[0]
        chord_reshape = np.lib.stride_tricks.as_strided(
            chord, shape=(num_seqs, num_steps), strides=(sss0 * seq_hop, sss0)
        )
        ssss0 = chord_change.strides[0]
        chord_change_reshape = np.lib.stride_tricks.as_strided(
            chord_change, shape=(num_seqs, num_steps), strides=(ssss0 * seq_hop, ssss0)
        )
        seq_len = np.array([num_steps for _ in range(num_seqs - 1)] + [num_steps - pad_size], dtype=np.int32)

        new_feature.append({
            "chroma": chroma_reshape,
            "tc": tc_reshape,
            "chord": chord_reshape,
            "chord_change": chord_change_reshape,
            "sequence_len": seq_len,
            "num_sequence": num_seqs
        })
    return new_feature
