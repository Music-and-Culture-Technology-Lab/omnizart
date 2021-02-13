import math

import numpy as np
import pretty_midi
from scipy.interpolate import interp1d

from omnizart.utils import get_logger
from omnizart.base import Label
from omnizart.constants.midi import LOWEST_MIDI_NOTE
from omnizart.constants.datasets import MusicNetStructure


logger = get_logger("Beat features")


def extract_feature_from_midi(midi_path, t_unit=0.01):
    """Extract feature for beat module from MIDI file.

    See Also
    --------
    omnizart.beat.features.extract_feature:
        The main feature extraction function of beat module.
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    labels = []
    for inst in midi.instruments:
        for note in inst.notes:
            labels.append(Label(start_time=note.start, end_time=note.end, note=note.pitch))
    return extract_feature(labels, t_unit=t_unit)


def extract_musicnet_feature(csv_path, t_unit=0.01):
    """Extract feature for beat module from MusicNet label file.

    See Also
    --------
    omnizart.beat.features.extract_feature:
        The main feature extraction function of beat module.
    """
    labels = MusicNetStructure.load_label(csv_path)
    return extract_feature(labels, t_unit=t_unit)


def extract_feature(labels, t_unit=0.01):
    """Extract feature representation required by beat module.

    Parameters
    ----------
    labels: list[Label]
        List of :class:`omnizart.base.Label` instances.
    t_unit: float
        Time unit of each frame of the output representation.

    Returns
    -------
    feature: 2D numpy array
        A piano roll like representation. Please refer to the original paper
        for more details.
    """
    max_sec = max(label.end_time for label in labels)
    frm_num = math.ceil(max_sec / t_unit)
    onset = np.zeros((frm_num, 88))
    dura = np.zeros((frm_num, 88))
    inter_onset_interval = np.zeros(frm_num)

    # Extract piano roll feature (onset, duration)
    onset_idx_set = set()
    for label in labels:
        on_idx = int(round(label.start_time / t_unit))
        off_idx = int(round(label.end_time / t_unit))
        pitch = int(label.note) - LOWEST_MIDI_NOTE
        onset[on_idx, pitch] = 1
        dura[on_idx:off_idx, pitch] = 1
        onset_idx_set.add(on_idx)

    # Extract IOI feature
    onset_idx_set = sorted(list(onset_idx_set))
    for idx, val in enumerate(onset_idx_set[1:], 1):
        sec_diff = (onset_idx_set[idx] - onset_idx_set[idx-1]) * t_unit  # noqa: E226
        inter_onset_interval[val] = sec_diff
    inter_onset_interval = np.expand_dims(inter_onset_interval, 1)

    # Extract spectral flux feature
    spec_flux = onset[1:] - onset[:-1]
    spec_flux[spec_flux < 0] = 0
    spec_flux = np.sum(spec_flux, axis=1)
    spec_flux = np.insert(spec_flux, 0, 0)
    spec_flux = np.expand_dims(spec_flux, 1)

    feature = np.concatenate([onset, dura, inter_onset_interval, spec_flux], axis=1)
    return feature


def _infer_beat_offset(labels, rounding=1):
    # Check there is really no integer beat first.
    for label in labels:
        if int(label.start_beat) == label.start_beat:
            return 0

    first_beat = labels[0].start_beat
    round_holder = 10 ** rounding
    return (first_beat * round_holder - int(first_beat * round_holder)) / round_holder


def extract_musicnet_label(csv_path, meter=4, t_unit=0.01, rounding=1, fade_out=15):
    """Label extraction function for MusicNet.

    This function extracts the beat and down beat information given the symbolic
    representations of MusicNet.

    Parameters
    ----------
    csv_path: Path
        Path to the ground-truth file in CSV format.
    meter: int
        Meter information of the piece. Currently it is default to the most common
        meter, which is 4. Since there is no meter information recorded in MusicNet,
        the meter value will always be 4 and apparently this is not always true.
    t_unit: int
        Time unit of each frame in seconds.
    rounding: int
        Round to position below decimal of start beat.
    fade_out: int
        Used to augment the sparse positive label in a fade-out manner, reducing
        the value from 1 to 1/fade_out, totaling in length of <fade_out>.
    """
    labels = MusicNetStructure.load_label(csv_path)

    # We found that some of the annotations in MusicNet may have a global beat offset,
    # making the whole piece lack of integer beats, and thus cause errors.
    # To adjust this, we retrieve the offset from the first note. And the offsets usually
    # start to occur from the second position below decimal.
    offset = _infer_beat_offset(labels, rounding=rounding)
    round_and_shift = lambda beat: round(beat - offset, rounding)

    # Initialize beat and down beat array.
    max_sec = max(label.end_time for label in labels)
    frm_num = math.ceil(max_sec / t_unit) + fade_out
    beat_arr = np.zeros(frm_num)
    down_beat_arr = np.zeros(frm_num)

    # Extract beat label data for future training.
    beat_idx_mapping = {}
    added_beats = []
    act_value = np.array([1 / (i + 1) for i in range(fade_out)])
    for label in labels:
        start_beat = round_and_shift(label.start_beat)
        if int(start_beat) != start_beat:
            # Not on the beat
            continue

        added_beats.append(start_beat)
        beat_idx = round(label.start_time / t_unit)
        beat_idx_mapping[start_beat] = beat_idx
        beat_arr[beat_idx:beat_idx + fade_out] = act_value
        if start_beat % meter == 1:
            down_beat_arr[beat_idx:beat_idx + fade_out] = act_value
    if len(beat_idx_mapping) == 0:
        logger.error("No integer beat found in the piece: %s", csv_path)

    # Recover missing beat position without time stamp by interpolation.
    max_beat = math.ceil(max(label.end_beat - offset for label in labels))
    min_beat = math.ceil(min(label.start_beat - offset for label in labels))
    beats = set(range(min_beat, max_beat + 1))
    missing_beats = beats - set(added_beats)
    interp_beat_idx = interp1d(
        sorted(beat_idx_mapping.keys()),
        sorted(beat_idx_mapping.values()),
        kind='linear',
        fill_value='extrapolate'
    )
    for beat in missing_beats:
        itp_beat_idx = int(interp_beat_idx(beat))
        if (itp_beat_idx + fade_out >= len(beat_arr)) or itp_beat_idx < 0:
            continue

        beat_arr[itp_beat_idx:itp_beat_idx + fade_out] = act_value
        if beat % meter == 1:
            down_beat_arr[itp_beat_idx:itp_beat_idx + fade_out] = act_value

    return beat_arr, down_beat_arr
