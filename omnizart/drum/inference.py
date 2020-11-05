import pretty_midi
import numpy as np
from scipy.signal import find_peaks


def get_3inst_ary(inst_13_ary_in):
    inst_3_ary_out = np.zeros_like(inst_13_ary_in)[:, :3]

    inst_3_ary_out[:, 0] = inst_13_ary_in[:, 0]
    inst_3_ary_out[:, 1] = inst_13_ary_in[:, 1]
    inst_3_ary_out[:, 2] = np.max([inst_13_ary_in[:, 4], inst_13_ary_in[:, 5], inst_13_ary_in[:, 6]], axis=0)
    return inst_3_ary_out


def inference(pred, m_beat_arr, bass_drum_th=0.85, snare_th=1.2, hihat_th=0.17):
    insts = get_3inst_ary(pred)

    norm = lambda x: (x - np.mean(x)) / np.std(x)
    bass_drum_act, _ = find_peaks(norm(insts[:, 0]), height=bass_drum_th, distance=1)
    snare_act, _ = find_peaks(norm(insts[:, 1]), height=snare_th, distance=1)
    hihat_act, _ = find_peaks(norm(insts[:, 2]), height=hihat_th, distance=1)

    drum_inst = pretty_midi.Instrument(program=1, is_drum=True, name="drums")

    def register_notes(act_list, pitch):
        for onset in m_beat_arr[act_list]:
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=onset, end=onset + 0.05)
            drum_inst.notes.append(note)

    register_notes(bass_drum_act, pitch=35)
    register_notes(snare_act, pitch=38)
    register_notes(hihat_act, pitch=42)

    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    midi.instruments.append(drum_inst)
    return midi
