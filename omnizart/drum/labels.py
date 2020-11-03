import pretty_midi
import numpy as np


def extract_label(label_path, m_beat_arr):
    """Extract drum label notes.

    Process ground-truth midi into numpy array representation.

    Parameters
    ----------
    label_path: Path
        Path to the midi file.
    m_beat_arr:
        Extracted mini-beat array of the coressponding audio piece.

    Returns
    -------
    drum_track_ary: numpy.ndarray
        The extracted label in numpy array. Should have a total of 128 classes
        of drum notes.

    See Also
    --------
    omnizart.feature.beat_for_drum.extract_mini_beat_from_audio_path:
        The function for extracting mini-beat array from the given audio path.
    """
    m_beat_range = []
    start = m_beat_arr[0] - (m_beat_arr[1] - m_beat_arr[0]) / 2
    end = m_beat_arr[0] + (m_beat_arr[1] - m_beat_arr[0]) / 2
    m_beat_range.append(start)
    m_beat_range.append(end)
    for idx, beat in enumerate(m_beat_arr[1:-1]):
        end = beat + (m_beat_arr[idx+1] - beat) / 2  # noqa: E226
        m_beat_range.append(end)
    end = m_beat_arr[-1] + (m_beat_arr[-1] - m_beat_arr[-2]) / 2
    m_beat_range.append(end)
    m_beat_range = np.array(m_beat_range)

    midi = pretty_midi.PrettyMIDI(label_path)
    notes = np.array([
        [nn.start, nn.pitch]
        for inst in midi.instruments
        for nn in inst.notes
        if inst.is_drum
    ])
    drum_track_ary = np.zeros([len(m_beat_arr), 128])
    for idx, beat in enumerate(m_beat_range[:-1]):
        for note in notes:
            if beat <= note[0] < m_beat_range[idx+1]:  # noqa: E226
                drum_track_ary[idx, int(note[1])] = 1.0
    return drum_track_ary


def extract_label_13_inst(label_path, m_beat_arr):
    """Extract 13 types of drum label notes.

    Process the MIDI drum notes into numpy array and concludes them
    into 13 different sub-classes of drum notes.

    Parameters
    ----------
    label_path: Path
        Path to the midi file.
    m_beat_arr:
        Extracted mini-beat array of the coressponding audio piece.

    Returns
    -------
    drum_track_ary: numpy.ndarray
        The extracted label in numpy array.

    See Also
    --------
    omnizart.drum.labels.extract_label:
        Complete drum label extraction with 128 output classes.
    omnizart.feature.beat_for_drum.extract_mini_beat_from_audio_path:
        The function for extracting mini-beat array from the given audio path.
    """
    label = extract_label(label_path, m_beat_arr)

    inst_ary_out = np.zeros([len(label), 13]).astype(np.float32)
    inst_ary_out[:, 0] = np.max(label[:, [33, 35, 36]], axis=1)  # Bass drum
    inst_ary_out[:, 1] = np.max(label[:, [27, 38, 40, 85, 87]], axis=1)  # Snare drum
    inst_ary_out[:, 2] = np.max(label[:, [37]], axis=1)  # Side Stick
    inst_ary_out[:, 3] = np.max(label[:, [39]], axis=1)  # Clap
    inst_ary_out[:, 4] = np.max(label[:, [42]], axis=1)  # Closed HH
    inst_ary_out[:, 5] = np.max(label[:, [44]], axis=1)  # Pedal HH
    inst_ary_out[:, 6] = np.max(label[:, [46]], axis=1)  # Open HH
    inst_ary_out[:, 7] = np.max(label[:, [41, 43]], axis=1)  # low-tom
    inst_ary_out[:, 8] = np.max(label[:, [45, 47]], axis=1)  # mid-tom
    inst_ary_out[:, 9] = np.max(label[:, [48, 50]], axis=1)  # high-tom
    inst_ary_out[:, 10] = np.max(label[:, [49, 55, 57]], axis=1)  # Crash
    inst_ary_out[:, 11] = np.max(label[:, [51, 53, 59]], axis=1)  # Ride
    inst_ary_out[:, 12] = np.max(label[:, [69, 70, 82]], axis=1)  # Maracas
    return label, inst_ary_out
