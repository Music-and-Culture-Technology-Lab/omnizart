import numpy as np
from PIL import Image

from omnizart.utils import LazyLoader


cfp = LazyLoader("cfp", globals(), "omnizart.feature.cfp")
cqt = LazyLoader("cqt", globals(), "omnizart.feature.cqt")
b4d = LazyLoader("b4d", globals(), "omnizart.feature.beat_for_drum")
hcfp = LazyLoader("hcfp", globals(), "omnizart.feature.hcfp")
chrom = LazyLoader("chrom", globals(), "omnizart.feature.chroma")


def extract_cfp_feature(audio_path, harmonic=False, harmonic_num=6, **kwargs):
    """Wrapper of CFP/HCFP feature extraction.

    Detailed available arguments can be found from the individual function.
    """
    if harmonic:
        spec, gcos, ceps, _ = hcfp.extract_hcfp(audio_path, harmonic_num=harmonic_num, **kwargs)
        return np.dstack([spec, gcos, ceps])

    z, spec, gcos, ceps, _ = cfp.extract_cfp(audio_path, **kwargs)
    return np.dstack([z.T, spec.T, gcos.T, ceps.T])


def get_frame_by_time(time_sec, sampling_rate=44100, hop_size=256):
    return int(round(time_sec * sampling_rate / hop_size))


def extract_patch_cqt(audio_path, sampling_rate=44100, hop_size=256):
    """Extract patched CQT feature.

    Leverages mini-beat information to determine the bound of each
    CQT patch.

    Parameters
    ----------
    audio_path: Path
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
    cqt_ext = cqt.extract_cqt(audio_path, sampling_rate=sampling_rate, a_hop=hop_size)
    mini_beat_arr = b4d.extract_mini_beat_from_audio_path(audio_path, sampling_rate=sampling_rate)

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
        m_beat_cqt_patch = Image.fromarray(m_beat_cqt_patch).resize((120, 120), resample=Image.BICUBIC)
        m_beat_cqt_patch_list.append(np.array(m_beat_cqt_patch))

    # convert cqt patch into cqt array
    return np.array(m_beat_cqt_patch_list), mini_beat_arr


def extract_chord_chroma(audio_path, segment_width=21, segment_hop=5, num_steps=100):
    _, chroma = chrom.extract_chroma(audio_path)

    pad_size = segment_width // 2
    chroma_pad = np.pad(chroma, ((pad_size, pad_size), (0, 0)), constant_values=0)
    segments = np.array([
        chroma_pad[i-pad_size:i+pad_size+1] for i in range(pad_size, pad_size+len(chroma), segment_hop)  # noqa:E226
    ])
    segments = segments.reshape([-1, segment_width * chroma.shape[1]])

    pad_size = 0 if len(segments)/num_steps == 0 else num_steps - len(segments)%num_steps  # noqa:E226,E228
    if pad_size != 0:
        segments = np.pad(segments, ((0, pad_size), (0, 0)), "constant", constant_values=0)

    feat_size = segments.shape[1]
    seq_hop = num_steps // 2
    num_seqs = int((len(segments) - num_steps) / seq_hop) + 1
    st0, st1 = segments.strides
    out = np.lib.stride_tricks.as_strided(
        segments, shape=(num_seqs, num_steps, feat_size), strides=(st0 * seq_hop, st0, st1)
    )
    return out
