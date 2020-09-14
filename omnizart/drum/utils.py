# pylint: disable=E1101
import numpy as np
import cv2


def get_frame_by_time(time_sec, sampling_rate=44100, hop_size=256):
    return int(round(time_sec * sampling_rate / hop_size))


def extract_patch_cqt(cqt_ext, mini_beat_arr, sampling_rate=44100, hop_size=256):
    """Extract patched CQT feature.

    Leverages mini-beat information to determine the bound of each
    CQT patch.

    Parameters
    ----------
    cqt_ext
        CQT feature of the audio.
    mini_beat_arr: 1D numpy array
        Mini-beat array, containing positions of mini-beats.

    Returns
    -------
    patch_cqt
        Extracted patch CQT feature.

    See Also
    --------
    omnizart.drum.transcribe: Entry point for transcribing drum.
    omnizart.feature.cqt.extract_cqt: Function for extracting CQT feature.
    omnizart.feature.beat_for_drum.extract_mini_beat_from_audio_path: Function for extracting mini-beat.
    """
    m_beat_cqt_patch_list = []
    for m_beat_t_cur in mini_beat_arr:
        # Get cqt patch data
        m_beat_t_cqt_range_stt = m_beat_t_cur - 0.200
        m_beat_t_cqt_range_end = m_beat_t_cur + 0.500
        m_beat_t_cqt_range_stt_w_offset = 1.0 + m_beat_t_cqt_range_stt
        m_beat_t_cqt_range_end_w_offset = 1.0 + m_beat_t_cqt_range_end
        m_beat_f_cqt_range_stt_w_offset = get_frame_by_time(
            m_beat_t_cqt_range_stt_w_offset, sampling_rate=sampling_rate, hop_size=hop_size
        )
        m_beat_f_cqt_range_end_w_offset = get_frame_by_time(
            m_beat_t_cqt_range_end_w_offset, sampling_rate=sampling_rate, hop_size=hop_size
        )

        m_beat_cqt_patch = cqt_ext[m_beat_f_cqt_range_stt_w_offset:m_beat_f_cqt_range_end_w_offset, :]
        m_beat_cqt_patch = cv2.resize(m_beat_cqt_patch, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
        m_beat_cqt_patch_list.append(m_beat_cqt_patch)

    # convert cqt patch into cqt array
    return np.array(m_beat_cqt_patch_list)


def create_batches(feature, mini_beat_per_seg, b_size=6):
    """Create a 4D input for model prediction.

    Parameters
    ----------
    feature: 3D numpy array
        Should be in shape [mini_beat_pos x time x freq].
    mini_beat_per_seg: int
        Number of mini beats in one segment (a beat).
    b_size: int
        Output batch size.

    Returns
    -------
    batch_feature: 5D numpy array
        Dimensions are [batches x b_size x time x freq x mini_beat_per_seg].
    """
    assert (len(feature.shape) == 3), f"Invalid feature shape: {feature.shape}. Should be three dimensional."

    # Pad zeros to the end of the feature if not long enough.
    if len(feature) < mini_beat_per_seg:
        pad_len = mini_beat_per_seg - len(feature)
        pads = np.zeros((pad_len, *feature.shape[1:]))
        feature = np.concatenate([feature, pads])

    hops = len(feature) - mini_beat_per_seg + 1
    hop_list = []
    for idx in range(hops):
        feat = feature[idx:idx + mini_beat_per_seg]
        hop_list.append(np.transpose(feat, axes=[1, 2, 0]))

    total_batches = int(np.ceil(len(hop_list) / b_size))
    batch_feature = []
    for idx in range(total_batches):
        batch_feature.append(hop_list[idx * b_size:(idx+1) * b_size])  # noqa: E226

    zero_feat = np.zeros_like(hop_list[0])
    pad_size = b_size - len(batch_feature[-1])
    for _ in range(pad_size):
        batch_feature[-1].append(zero_feat)

    return np.array(batch_feature)


def merge_batches(batch_pred):
    """Reverse process of create_batches.

    Merges a 5D batched-prediction into 2D output.
    """
    assert(len(batch_pred.shape) == 5)
    assert(batch_pred.shape[-1] == 1)

    batches, b_size, out_classes, mini_beat_per_seg = batch_pred.shape[:4]
    pred = np.zeros((batches*b_size + mini_beat_per_seg - 1, out_classes))  # noqa: E226
    for b_idx, batch in enumerate(batch_pred):
        for s_idx, step in enumerate(batch):
            start_idx = b_idx*b_size + s_idx  # noqa: E226
            end_idx = start_idx + mini_beat_per_seg
            pred[start_idx:end_idx] += step.T.squeeze()

    max_len = min(mini_beat_per_seg - 1, len(pred) - mini_beat_per_seg)
    pred[max_len:-max_len] /= max_len + 1
    for idx in range(max_len):
        pred[idx] /= idx + 1
        pred[-1 - idx] /= idx + 1
    return pred
