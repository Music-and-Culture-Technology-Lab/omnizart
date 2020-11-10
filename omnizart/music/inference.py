# pylint: disable=W0102,R0914

import math

import pretty_midi
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from librosa import note_to_midi

from omnizart.constants.midi import MUSICNET_INSTRUMENT_PROGRAMS, MIDI_PROGRAM_NAME_MAPPING
from omnizart.utils import get_logger


logger = get_logger("Music Inference")


def roll_down_sample(data, occur_num=3, base=88):
    """Down sample feature size for a single pitch.

    Down sample the feature size from 354 to 88 for infering the notes.

    Parameters
    ----------
    data: 2D numpy array
        The thresholded 2D prediction..
    occur_num: int
        For each pitch, the original prediction expands 4 bins wide. This value determines how many positive bins
        should there be to say there is a real activation after down sampling.
    base
        Should be constant as there are 88 pitches on the piano.

    Returns
    -------
    return_v: 2D numpy array
        Down sampled prediction.

    Warnings
    --------
    The parameter `data` should be thresholded!
    """

    total_roll = data.shape[1]
    assert total_roll % base == 0, f"Wrong length: {total_roll}, {total_roll} % {base} should be zero!"

    scale = round(total_roll / base)
    assert 0 < occur_num <= scale

    return_v = np.zeros((len(data), base), dtype=int)

    for i in range(0, data.shape[1], scale):
        total = np.sum(data[:, i:i + scale], axis=1)
        return_v[:, int(i / scale)] = np.where(total >= occur_num, total / occur_num, 0)
    return_v = np.where(return_v >= 1, 1, return_v)

    return return_v


def down_sample(pred, occur_num=3):
    """Down sample multi-channel predictions along the feature dimension.

    Down sample the feature size from 354 to 88 for infering the notes from a multi-channel prediction.

    Parameters
    ----------
    pred: 3D numpy array
        Thresholded prediction with multiple channels. Dimension: [timesteps x pitch x instruments]
    occur_num: int
        Minimum occurance of each pitch for determining true activation of the pitch.

    Returns
    -------
    d_sample: 3D numpy array
        Down-sampled prediction. Dimension: [timesteps x 88 x instruments]
    """
    d_sample = roll_down_sample(pred[:, :, 0], occur_num=occur_num)
    for i in range(1, pred.shape[2]):
        d_sample = np.dstack([d_sample, roll_down_sample(pred[:, :, i], occur_num=occur_num)])

    return d_sample


def infer_pitch(pitch, shortest=5, offset_interval=6):
    w_on = pitch[:, 2]
    w_dura = pitch[:, 1]

    peaks, _ = find_peaks(w_on, distance=shortest, width=5)
    if len(peaks) == 0:
        return []

    notes = []
    adjust = 5 if shortest == 10 else 2
    for i in range(len(peaks) - 1):
        notes.append({"start": peaks[i] - adjust, "end": peaks[i + 1] - adjust, "stren": pitch[peaks[i], 2]})
    notes.append({"start": peaks[-1] - adjust, "end": len(w_on) - adjust, "stren": pitch[peaks[-1], 2]})

    del_idx = []
    for idx, peak in enumerate(peaks):
        upper = int(peaks[idx + 1]) if idx < len(peaks) - 1 else len(w_dura)
        for i in range(peak, upper):
            if np.sum(w_dura[i:i + offset_interval]) == 0:
                if i - notes[idx]["start"] - adjust < shortest - 1:
                    del_idx.append(idx)
                else:
                    notes[idx]["end"] = i - adjust
                break

    for ii, i in enumerate(del_idx):
        del notes[i - ii]

    return notes


def infer_piece(piece, shortest_sec=0.1, offset_sec=0.12, t_unit=0.02):
    """
        Dim: time x 88 x 4 (off, dura, onset, offset)
    """
    assert piece.shape[1] == 88, "Please down sample the pitch to 88 first (current: {}).format(piece.shape[1])"
    min_align_diff = 1  # to align the onset between notes with a short time difference

    notes = []
    for i in range(88):
        print("Pitch: {}/{}".format(i + 1, 88), end="\r")

        pitch = piece[:, i]
        if np.sum(pitch) <= 0:
            continue

        pns = infer_pitch(pitch, shortest=round(shortest_sec / t_unit), offset_interval=round(offset_sec / t_unit))
        for note in pns:
            note["pitch"] = i
            notes.append(note)
    print(" " * 80, end="\r")

    notes = sorted(notes, key=lambda d: d["start"])
    last_start = 0
    for note in notes:
        start_diff = note["start"] - last_start
        if start_diff < min_align_diff:
            note["start"] -= start_diff
            note["end"] -= start_diff
        else:
            last_start = note["start"]

    return notes


def find_min_max_stren(notes):
    """Function for detemine the note velocity accroding to prediction value.

    Parameters
    ----------
    notes: list[dict]
        Data structure returned by function `infer_piece`.
    """
    stren = [nn["stren"] for nn in notes] + [0.5]
    return np.min(stren), np.max(stren)


def find_occur(pitch, t_unit=0.02, min_duration=0.03):
    """Find the onset and offset of a thresholded prediction.

    Parameters
    ----------
    pitch: 1D numpy array
        Time series of predicted pitch activations.
    t_unit: float
        Time unit of each entry.
    min_duration: float
        Minimum interval of each note in seconds.
    """

    min_duration = max(t_unit, min_duration)
    min_frm = max(0.1, min_duration/t_unit - 1)  # noqa: E226

    cand = np.where(pitch > 0.5)[0]
    if len(cand) == 0:
        return []

    start = cand[0]
    last = cand[0]
    note = []
    for cidx in cand:
        if cidx - last > 1:
            if last - start >= min_frm:
                note.append({"onset": start, "offset": last})
            start = cidx
        last = cidx

    if last - start >= min_frm:
        note.append({"onset": start, "offset": last})
    return note


def to_midi(notes, t_unit=0.02):
    """Translate the intermediate data into final output MIDI file."""

    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    # Some tricky steps to determine the velocity of the notes
    l_bound, u_bound = find_min_max_stren(notes)
    s_low = 60
    s_up = 127
    v_map = lambda stren: int(
        s_low + ((s_up-s_low) * ((stren-l_bound) / (u_bound-l_bound+0.0001)))  # noqa: E226
    )

    low_b = note_to_midi("A0")
    coll = set()
    for note in notes:
        pitch = note["pitch"] + low_b
        start = note["start"] * t_unit
        end = note["end"] * t_unit
        volume = v_map(note["stren"])
        coll.add(pitch)
        m_note = pretty_midi.Note(velocity=volume, pitch=pitch, start=start, end=end)
        piano.notes.append(m_note)
    midi.instruments.append(piano)
    return midi


def interpolation(data, ori_t_unit=0.02, tar_t_unit=0.01):
    """Interpolate between each frame to increase the time resolution.

    The default setting of feature extraction has time resolution of 0.02 seconds for each frame.
    To fit the conventional evaluation settings, which has time resolution of 0.01 seconds, we additionally
    apply the interpolation function to increase time resolution. Here we use `Cubic Spline` for the
    estimation.
    """
    assert len(data.shape) == 2

    ori_x = np.arange(len(data))
    tar_x = np.arange(0, len(data), tar_t_unit / ori_t_unit)
    func = CubicSpline(ori_x, data, axis=0)
    return func(tar_x)


def norm(data):
    return (data - np.mean(data)) / np.std(data)


def norm_onset_dura(pred, onset_th, dura_th, interpolate=True, normalize=True):
    """Normalizes prediction values of onset and duration channel."""

    length = len(pred) * 2 if interpolate else len(pred)
    norm_pred = np.zeros((length, ) + pred.shape[1:])
    onset = interpolation(pred[:, :, 2])
    dura = interpolation(pred[:, :, 1])

    onset = np.where(onset < dura, 0, onset)
    norm_onset = norm(onset) if normalize else onset
    onset = np.where(norm_onset < onset_th, 0, norm_onset)
    norm_pred[:, :, 2] = onset

    norm_dura = norm(dura) + onset if normalize else dura + onset
    dura = np.where(norm_dura < dura_th, 0, norm_dura)
    norm_pred[:, :, 1] = dura

    return norm_pred


def norm_split_onset_dura(pred, onset_th, lower_onset_th, split_bound, dura_th, interpolate=True, normalize=True):
    """An advanced version of function for normalizing onset and duration channel.

    From the extensive experiments, we observe that the average prediction value for high and low frequency are
    different. Lower pitches tend to have smaller values, while higher pitches having larger. To acheive better
    transcription results, the most straight-forward solution is to assign different thresholds for low and
    high frequency part. And this is what this function provides for the purpose.

    Parameters
    ----------
    pred
        The predictions.
    onset_th: float
        Threshold for high frequency part.
    lower_onset_th: float
        Threshold for low frequency part.
    split_bound: int
        The split point of low and high frequency part. Value should be within 0~87.
    interpolate: bool
        Whether to apply interpolation between each frame to increase time resolution.
    normalize: bool
        Whether to normalize the prediction values.

    Returns
    -------
    pred
        Thresholded prediction, having value either 0 or 1.
    """

    upper_range = range(4 * split_bound, 352)
    upper_pred = pred[:, upper_range]
    upper_pred = norm_onset_dura(upper_pred, onset_th, dura_th, interpolate=interpolate, normalize=normalize)

    lower_range = range(4 * split_bound)
    lower_pred = pred[:, lower_range]
    lower_pred = norm_onset_dura(lower_pred, lower_onset_th, dura_th, interpolate=interpolate, normalize=normalize)

    return np.hstack([lower_pred, upper_pred])


def threshold_type_converter(threshold, length):
    """Convert scalar value to a list with the same value."""
    if isinstance(threshold, list):
        assert len(threshold) == length
    else:
        threshold_list = [threshold for _ in range(length)]
    return threshold_list


def entropy(data, bins=200):
    min_v = -20
    max_v = 30
    interval = (max_v-min_v) / bins  # noqa: E226
    cut_offs = [min_v + i*interval for i in range(bins + 1)]  # noqa: E226
    discrete_v = np.digitize(data, cut_offs)
    _, counts = np.unique(discrete_v, return_counts=True)
    probs = counts / np.sum(counts)
    ent = 0
    for prob in probs:
        ent -= prob * math.log(prob, math.e)

    return ent


def note_inference(
    pred,
    mode="note",
    onset_th=7.5,
    lower_onset_th=None,
    split_bound=36,
    dura_th=2,
    frm_th=1,
    normalize=True,
    t_unit=0.02,
):
    if "note" in mode:
        if lower_onset_th is not None:
            norm_pred = norm_split_onset_dura(
                pred,
                onset_th=onset_th,
                lower_onset_th=lower_onset_th,
                split_bound=split_bound,
                dura_th=dura_th,
                interpolate=True,
                normalize=normalize,
            )
        else:
            norm_pred = norm_onset_dura(pred, onset_th=onset_th, dura_th=dura_th, interpolate=True, normalize=normalize)

        norm_pred = np.where(norm_pred > 0, norm_pred + 1, 0)
        notes = infer_piece(down_sample(norm_pred), t_unit=0.01)
        midi = to_midi(notes, t_unit=t_unit / 2)

    else:
        ch_num = pred.shape[2]
        if ch_num == 2:
            mix = pred[:, :, 1]
        elif ch_num == 3:
            mix = (pred[:, :, 1] + pred[:, :, 2]) / 2
        else:
            raise ValueError("Unknown channel length: {}".format(ch_num))

        prob = norm(mix) if normalize else mix
        prob = np.where(prob > frm_th, 1, 0)
        prob = roll_down_sample(prob)

        notes = []
        for idx in range(prob.shape[1]):
            p_note = find_occur(prob[:, idx], t_unit=t_unit)
            for note in p_note:
                note_info = {
                    "pitch": idx,
                    "start": note["onset"],
                    "end": note["offset"],
                    "stren": mix[int(note["onset"] * t_unit), idx * 4],
                }
                notes.append(note_info)
        midi = to_midi(notes, t_unit=t_unit)
    return midi


def multi_inst_note_inference(
    pred,
    mode="note-stream",
    onset_th=5,
    dura_th=2,
    frm_th=1,
    inst_th=0.95,
    normalize=True,
    t_unit=0.02,
    channel_program_mapping=MUSICNET_INSTRUMENT_PROGRAMS,
):
    """Function for infering raw multi-instrument predictions.

    Parameters
    ----------
    mode: {'note-stream', 'note', 'frame-stream', 'frame'}
        Inference mode.
        Difference between 'note' and 'frame' is that the former consists of two note attributes, which are 'onset' and
        'duration', and the later only contains 'duration', which in most of the cases leads to worse listening
        experience.
        With postfix 'stream' refers to transcribe instrument at the same time, meaning classifying each notes into
        instrument classes, or says different tracks.
    onset_th: float
        Threshold of onset channel. Type of list or float
    dura_th: float
        Threshold of duration channel. Type of list or float
    inst_th: float
        Threshold of deciding a instrument is present or not according to Std. of prediction.
    normalize: bool
        Whether to normalize the predictions. For more details, please refer to our
        `paper <https://bit.ly/2QhdWX5>`_
    t_unit: float
        Time unit for each frame. Should not be modified unless you have different settings during the feature
        extraction
    channel_program_mapping: list[int]
        Mapping prediction channels to MIDI program numbers.

    Returns
    -------
    out_midi
        A pretty_midi.PrettyMIDI object.

    References
    ----------
    Publications can be found `here <https://bit.ly/2QhdWX5>`_.
    """

    if mode in ["note-stream", "note", "pop-note-stream"]:
        ch_per_inst = 2
    elif mode in ["frame-stream", "frame"]:
        ch_per_inst = 2
    elif mode in ["true-frame", "true-frame-stream"]:
        # For older version compatibility that models were trained on pure frame-level.
        mode = "frame"
        ch_per_inst = 1
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    assert (pred.shape[-1] - 1) % ch_per_inst == 0, f"Input shape: {pred.shape}"

    ch_container = []
    iters = (pred.shape[-1] - 1) // ch_per_inst
    for i in range(ch_per_inst):
        # First item would be duration channel
        # Second item would be onset channel
        item = pred[:, :, [it*ch_per_inst + i + 1 for it in range(iters)]]  # noqa: E226
        ch_container.append(norm(item) if normalize else item)

    if not mode.endswith("-stream") and mode != "true_frame":
        # Some different process for none-instrument care cases
        # Merge all channels into first channel
        iters = 1
        for i in range(ch_per_inst):
            normed_p = ch_container[i]
            normed_p[:, :, 0] = np.average(normed_p, axis=2)
            ch_container[i] = normed_p

    # Handling given thresholds that could be type of either scalar value or list
    onset_th = threshold_type_converter(onset_th, iters)
    dura_th = threshold_type_converter(dura_th, iters)
    frm_th = threshold_type_converter(frm_th, iters)

    # Multi-instrument inference loop, iterate through different instrument channels
    zeros = np.zeros((pred.shape[:-1]))
    out_midi = pretty_midi.PrettyMIDI()
    for i in range(iters):
        normed_ch = []
        std = 0
        ent = 0
        # Compute confidence of the instrument
        for ii in range(ch_per_inst):
            cha = ch_container[ii][:, :, i]
            std += np.std(cha)
            ent += entropy(cha)
            normed_ch.append(cha)

        confidence = "std: {:.3f} ent: {:.3f} mult: {:.3f}".format(
            std / ch_per_inst, ent / ch_per_inst, std * ent / ch_per_inst**2
        )
        logger.debug("Instrument confidence: %s", confidence)
        if iters > 1 and (std / ch_per_inst < inst_th):
            # Filter out instruments that the confidence is under the given threshold
            continue

        # Infer notes according to raw predictions
        normed_p = np.dstack([zeros] + normed_ch)
        midi = note_inference(
            normed_p,
            mode=mode,
            onset_th=onset_th[i],
            dura_th=dura_th[i],
            frm_th=frm_th[i],
            normalize=normalize,
            t_unit=t_unit,
        )

        # Assign instrument class to the infered MIDI accroding to its channel index
        inst_program = channel_program_mapping[i]
        inst_name = MIDI_PROGRAM_NAME_MAPPING[str(inst_program)]
        inst = pretty_midi.Instrument(program=inst_program, name=inst_name)
        inst.notes = midi.instruments[0].notes
        out_midi.instruments.append(inst)

    return out_midi
