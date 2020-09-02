"""Customized feature extraction function.

Combined CQT and beat tracking information as the input feature.
After consideration, we decide not to put this function under
`feature/` folder because of its coupling between CQT and beat
info.

"""

# pylint: disable=E1101

import numpy as np
import cv2

from omnizart.drum.utils import get_frame_by_time


def extract_patch_cqt(cqt_ext, mini_beat_arr):
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
        m_beat_f_cqt_range_stt_w_offset = get_frame_by_time(m_beat_t_cqt_range_stt_w_offset)
        m_beat_f_cqt_range_end_w_offset = get_frame_by_time(m_beat_t_cqt_range_end_w_offset)

        m_beat_cqt_patch = cqt_ext[m_beat_f_cqt_range_stt_w_offset:m_beat_f_cqt_range_end_w_offset, :]
        m_beat_cqt_patch = cv2.resize(m_beat_cqt_patch, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
        m_beat_cqt_patch_list.append(m_beat_cqt_patch)

    # convert cqt patch into cqt array
    return np.array(m_beat_cqt_patch_list)
