import numpy as np


def _conv(seq, window):
    half_len = len(window) // 2
    end_idx = len(seq) - half_len
    total = sum(window)
    container = []
    for val in seq[:half_len]:
        container.append(val)
    for idx in range(half_len, end_idx):
        container.append(np.dot(seq[idx-half_len:idx+half_len+1], window) / total)  # noqa: E226
    for val in seq[-half_len:]:
        container.append(val)
    return np.array(container)


def _find_peaks(seq, ctx_len=2, threshold=0.5):
    # Discard the first and the last <ctx_len> frames.
    peaks = []
    for idx in range(ctx_len, len(seq) - ctx_len - 1):
        cur_val = seq[idx]
        if cur_val < threshold:
            continue
        if not all(cur_val > seq[idx-ctx_len:idx]):
            continue
        if not all(cur_val >= seq[idx+1:idx+ctx_len+1]):
            continue
        peaks.append(idx)
    return peaks


def _find_first_bellow_th(seq, threshold=0.5):
    activate = False
    for idx, val in enumerate(seq):
        if val > threshold:
            activate = True
        if activate and val < threshold:
            return idx
    return 0


def infer_interval_original(pred, ctx_len=2, threshold=0.5, t_unit=0.02):
    """Original implementation of interval inference.

    After checking the inference results of this implementation, we found
    there are lots of missing notes that aren't in the inferenced results.
    This function is just leaving for reference.

    Parameters
    ----------
    pred:
        Raw prediction array.
    ctx_len: int
        Context length for determing peaks.
    threhsold: float
        Threshold for prediction values to be taken as true positive.
    t_unit: float
        Time unit of each frame.

    Returns
    -------
    interval: list[tuple[float, float]]
        Pairs of inferenced onset and offset time in seconds.
    """
    dura_seq = pred[:, 0]
    onset_seq = pred[:, 2]
    offset_seq = pred[:, 4]

    window = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    onset_seq = _conv(onset_seq, window)
    offset_seq = _conv(offset_seq, window)

    on_peaks = _find_peaks(onset_seq, ctx_len=ctx_len, threshold=threshold)
    off_peaks = _find_peaks(offset_seq, ctx_len=ctx_len, threshold=threshold)
    if len(on_peaks) == 0 or len(off_peaks) == 0:
        return None

    # Clearing out offsets before first onset (since onset is more accurate)
    off_peaks = [idx for idx in off_peaks if idx > on_peaks[0]]

    mix_peaks = sorted(on_peaks + off_peaks)
    tidx = 0
    est_intervals = []
    while tidx < len(mix_peaks):
        if tidx == len(mix_peaks) - 1:
            break
        if tidx == 0 and mix_peaks[tidx] not in on_peaks:
            tidx += 1

        if mix_peaks[tidx in on_peaks] and mix_peaks[tidx + 1] in off_peaks:
            if mix_peaks[tidx] == mix_peaks[tidx + 1]:
                tidx += 1
                continue
            if mix_peaks[tidx + 1] > mix_peaks[tidx] + 1:
                est_intervals.append((mix_peaks[tidx] * t_unit, mix_peaks[tidx + 1] * t_unit))
            assert mix_peaks[tidx] < mix_peaks[tidx + 1]
            tidx += 2
        elif mix_peaks[tidx] in on_peaks and mix_peaks[tidx + 1] in on_peaks:
            dura_slice = dura_seq[mix_peaks[tidx]:mix_peaks[tidx + 1]]
            off_idx = _find_first_bellow_th(dura_slice) + mix_peaks[tidx]
            if off_idx != mix_peaks[tidx] and off_idx > mix_peaks[tidx] + 1:
                est_intervals.append((mix_peaks[tidx] * t_unit, off_idx * t_unit))
                assert mix_peaks[tidx] < off_idx
            tidx += 1
        elif mix_peaks[tidx] in off_peaks:
            tidx += 1

    return np.array(est_intervals)


def infer_interval(pred, ctx_len=2, threshold=0.5, min_dura=0.1, t_unit=0.02):
    """Improved version of interval inference function.

    Inference the onset and offset time of notes given the raw prediction values.

    Parameters
    ----------
    pred:
        Raw prediction array.
    ctx_len: int
        Context length for determing peaks.
    threhsold: float
        Threshold for prediction values to be taken as true positive.
    min_dura: float
        Minimum duration for a note, in seconds.
    t_unit: float
        Time unit of each frame.

    Returns
    -------
    interval: list[tuple[float, float]]
        Pairs of inferenced onset and offset time in seconds.
    """
    on_peaks = _find_peaks(pred[:, 2], ctx_len=ctx_len, threshold=threshold)
    off_peaks = _find_peaks(pred[:, 4], ctx_len=ctx_len, threshold=threshold)
    if len(on_peaks) == 0 or len(off_peaks) == 0:
        return None

    # Clearing out offsets before first onset (since onset is more accurate)
    off_peaks = [idx for idx in off_peaks if idx > on_peaks[0]]

    on_peak_id = 0
    est_interval = []
    min_len = min_dura / t_unit
    while on_peak_id < len(on_peaks) - 1:
        on_id = on_peaks[on_peak_id]
        next_on_id = on_peaks[on_peak_id + 1]

        off_peak_id = np.where(np.array(off_peaks) >= on_id + min_len)[0][0]
        off_id = off_peaks[off_peak_id]
        if on_id < next_on_id < off_id \
            and np.mean(pred[on_id:next_on_id, 1]) > np.mean(pred[on_id:next_on_id, 0]):
            # Discard current onset, since the duration between current and
            # next onset shows an inactive status.
            on_peak_id += 1
            continue

        if off_id > next_on_id:
            # Missing offset between current and next onset.
            if (off_id - next_on_id) < min_len:
                # Assign the offset after the next onset to the current onset.
                est_interval.append((on_id * t_unit, off_id * t_unit))
                on_peak_id += 1
            else:
                # Insert an additional offset.
                est_interval.append((on_id * t_unit, next_on_id * t_unit))
                on_peak_id += 1
        elif (off_id - on_id) >= min_len:
            # Normal case that one onset has a corressponding offset.
            est_interval.append((on_id * t_unit, off_id * t_unit))
            on_peak_id += 1
        else:
            # Do nothing
            on_peak_id += 1

    # Deal with the border case, the last onset peak.
    on_id = on_peaks[-1]
    off_id = _find_first_bellow_th(pred[on_id:, 0], threshold=threshold) + on_id
    if off_id - on_id >= min_len:
        est_interval.append((on_id * t_unit, off_id * t_unit))

    return np.array(est_interval)


if __name__ == "__main__":
    pred = np.load("/data/omnizart/omnizart/test.npy")
    data = np.copy(pred)
    interval = infer_interval(pred)
    for on, off in interval:
        data[int(round(on*50)), :3] = 0
        data[int(round(off*50)), 3:] = 0
