import os
import tempfile

import pretty_midi
import numpy as np
import scipy.io.wavfile as wave

from omnizart.feature.beat_for_drum import extract_mini_beat_from_audio_path
from omnizart.constants.midi import SOUNDFONT_PATH
from omnizart.utils import ensure_path_exists


TMP_WAV_DIR = tempfile.mktemp()


def synth_midi(midi_path, sampling_rate=44100, out_path=TMP_WAV_DIR):
    midi = pretty_midi.PrettyMIDI(midi_path)
    raw_wav = midi.fluidsynth(fs=sampling_rate, sf2_path=SOUNDFONT_PATH)
    if out_path is not None:
        filename = os.path.basename(midi_path).replace(".mid", ".wav")
        ensure_path_exists(out_path)
        out_path = os.path.join(out_path, filename)
        wave.write(out_path, sampling_rate, raw_wav)
        return out_path

    wave.write(midi_path.replace(".mid", ".wav"), sampling_rate, raw_wav)
    return midi_path.replace(".mid", ".wav")


def extract_label(label_path, m_beat_arr, sampling_rate=22050):
    m_beat_range = []
    start = m_beat_arr[0] - (m_beat_arr[1] - m_beat_arr[0]) / 2
    end = m_beat_arr[0] + (m_beat_arr[1] - m_beat_arr[0]) / 2
    m_beat_range.append(start)
    m_beat_range.append(end)
    for idx, beat in enumerate(m_beat_arr[1:-1]):
        end = beat + (m_beat_arr[idx+1] - beat) / 2
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
            if beat <= note[0] < m_beat_range[idx+1]:
                drum_track_ary[idx, int(note[1])] = 1.0
    return drum_track_ary


def extract_label_13_inst(label_path, m_beat_arr, sampling_rate=22050):
    label = extract_label(label_path, m_beat_arr, sampling_rate=sampling_rate)

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
