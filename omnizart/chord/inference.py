import csv

import pretty_midi
import numpy as np

from omnizart.constants.feature import CHORD_INT_MAPPING


C_MAJ_TEN_DEGREE = np.array([36, 43, 52], dtype=np.int32)
C_MIN_TEN_DEGREE = np.array([36, 43, 51], dtype=np.int32)
CHORD_MIDI_NOTES = {
    "C:maj": C_MAJ_TEN_DEGREE,
    "C#:maj": C_MAJ_TEN_DEGREE + 1,
    "D:maj": C_MAJ_TEN_DEGREE + 2,
    "D#:maj": C_MAJ_TEN_DEGREE + 3,
    "E:maj": C_MAJ_TEN_DEGREE + 4,
    "F:maj": C_MAJ_TEN_DEGREE + 5,
    "F#:maj": C_MAJ_TEN_DEGREE + 6,
    "G:maj": C_MAJ_TEN_DEGREE + 7,
    "G#:maj": C_MAJ_TEN_DEGREE + 8,
    "A:maj": C_MAJ_TEN_DEGREE + 9,
    "A#:maj": C_MAJ_TEN_DEGREE + 10,
    "B:maj": C_MAJ_TEN_DEGREE + 11,
    "C:min": C_MIN_TEN_DEGREE,
    "C#:min": C_MIN_TEN_DEGREE + 1,
    "D:min": C_MIN_TEN_DEGREE + 2,
    "D#:min": C_MIN_TEN_DEGREE + 3,
    "E:min": C_MIN_TEN_DEGREE + 4,
    "F:min": C_MIN_TEN_DEGREE + 5,
    "F#:min": C_MIN_TEN_DEGREE + 6,
    "G:min": C_MIN_TEN_DEGREE + 7,
    "G#:min": C_MIN_TEN_DEGREE + 8,
    "A:min": C_MIN_TEN_DEGREE + 9,
    "A#:min": C_MIN_TEN_DEGREE + 10,
    "B:min": C_MIN_TEN_DEGREE + 11,
}


def inference(chord_pred, t_unit, min_dura=0.1):
    no_chord = CHORD_INT_MAPPING["N"]
    chord_pred = np.pad(chord_pred, (1, 1), constant_values=no_chord)
    chord_change = np.where(chord_pred[:-1] != chord_pred[1:])[0]
    rev_map = {v: k for k, v in CHORD_INT_MAPPING.items()}
    info = []
    notes = []
    last_chord_name = "N"
    for idx, ch_idx in enumerate(chord_change[1:], 1):
        chord_num = chord_pred[ch_idx]
        chord_name = rev_map[chord_num]
        if chord_name in ["X", "N"]:
            continue

        start_t = (chord_change[idx-1] + 1) * t_unit  # noqa: E226
        end_t = ch_idx * t_unit
        if end_t - start_t >= min_dura:
            last_chord_name = chord_name
            info.append({
                "chord": chord_name,
                "start": start_t,
                "end": end_t
            })
            for pitch in CHORD_MIDI_NOTES[chord_name]:
                notes.append(pretty_midi.Note(start=start_t, end=end_t, pitch=pitch, velocity=100))
        elif last_chord_name not in ["X", "N"]:
            # Duration of the current chord shorter than expected.
            # Append the activation to the last chord.
            info[-1]["end"] = end_t
            note_count = len(CHORD_MIDI_NOTES[last_chord_name])
            for lidx in range(-note_count, 0):
                notes[lidx].end = end_t

    inst = pretty_midi.Instrument(program=0)
    inst.notes += notes
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.append(inst)
    return midi, info


def write_csv(info, output="./chord.csv"):
    with open(output, "w", newline='') as out:
        fieldnames = ["chord", "start", "end"]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for chord in info:
            writer.writerow(chord)
