import cv2
import numpy as np

from omnizart.feature.cfp import extract_cfp
from omnizart.feature.hcfp import extract_hcfp
from omnizart.feature.cqt import extract_cqt
from omnizart.feature.beat_for_drum import extract_mini_beat_from_audio_path


def extract_cfp_feature(audio_path, harmonic=False, harmonic_num=6, **kwargs):
    """Wrapper of CFP/HCFP feature extraction.

    Detailed available arguments can be found from the individual function.
    """
    if harmonic:
        spec, gcos, ceps, _ = extract_hcfp(audio_path, harmonic_num=harmonic_num, **kwargs)
        return np.dstack([spec, gcos, ceps])

    z, spec, gcos, ceps, _ = extract_cfp(audio_path, **kwargs)
    return np.dstack([z.T, spec.T, gcos.T, ceps.T])


def get_frame_by_time(time_sec, sampling_rate=44100, hop_size=256):
    return int(round(time_sec * sampling_rate / hop_size))


def extract_patch_cqt(input_audio, sampling_rate=44100, hop_size=256):
    """Extract patched CQT feature.

    Leverages mini-beat information to determine the bound of each
    CQT patch.

    Parameters
    ----------
    input_audio: Path
        Path to the wav file.

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
    cqt_ext = extract_cqt(input_audio)
    mini_beat_arr = extract_mini_beat_from_audio_path(input_audio)

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
    return np.array(m_beat_cqt_patch_list), mini_beat_arr
