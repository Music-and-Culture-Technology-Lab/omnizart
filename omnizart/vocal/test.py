import numpy as np
import pretty_midi

from omnizart.vocal.inference import infer_interval, infer_interval_original


def infer_midi():
    pred = np.load("/data/omnizart/omnizart/test_ava.npy")
    #interval = infer_interval_original(pred)
    interval = infer_interval(pred)
    contour_path = "/data/omnizart/omnizart/01-D_AMairena_f0.txt"
    contour_path = "/data/omnizart/omnizart/Ava_f0.txt"
    lines = open(contour_path, "r").readlines()
    lines = np.array([float(line.strip()) for line in lines])
    notes = []
    skip_num = 0
    for onset, offset in interval:
        start_idx = int(round(onset*50))
        end_idx = int(round(offset*50))
        freqs = lines[start_idx:end_idx]
        freqs = freqs[freqs > 0]
        if len(freqs) == 0:
            print("Skipped")
            skip_num += 1
            continue
        avg_hz = freqs[len(freqs)//2]
        # avg_hz = np.mean(freqs)
        note_num = int(round(pretty_midi.hz_to_note_number(avg_hz)))
        note = pretty_midi.Note(velocity=80, pitch=note_num, start=onset, end=offset)
        notes.append(note)
    inst = pretty_midi.Instrument(program=0)
    inst.notes += notes
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.append(inst)
    print(skip_num)
    return midi


if __name__ == "__main__":
    midi = infer_midi()
    midi.write("mud_vocal.mid")
