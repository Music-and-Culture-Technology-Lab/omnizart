import pretty_midi
from scipy.signal import find_peaks


BEAT_NOTE_NUM = 42  # Hihat
DOWN_BEAT_NOTE_NUM = 36  # Bass drum


def inference(pred, beat_th=0.5, down_beat_th=0.5, min_dist=0.3, t_unit=0.1):
    """Infers the beat and down beat positions from the raw prediction values.

    Parameters
    ----------
    pred: 2D numpy array
        The prediction of the model.
    beat_th: float
        Threshold for beat channel.
    down_beat_th: float
        Threshold for down beat channel.
    min_dist: float
        Minimum distance between two beat positions in seconds.
    t_unit: float
        Time unit of each frame in seconds.

    Returns
    -------
    midi: pretty_midi.PrettyMIDI
        Inferred beat positions recorded as MIDI notes. Information of beat
        and down beat are recorded in two different instrument tracks.
    """
    mdist = max(1, round(min_dist / t_unit))
    beat_pos, _ = find_peaks(pred[:, 0], height=beat_th, distance=mdist)
    db_pos, _ = find_peaks(pred[:, 1], height=down_beat_th, distance=mdist)

    beat_notes = []
    for pos in beat_pos:
        start_time = pos * t_unit
        end_time = start_time + 0.5
        beat_notes.append(
            pretty_midi.Note(start=start_time, end=end_time, pitch=BEAT_NOTE_NUM, velocity=100)
        )

    db_notes = []
    for pos in db_pos:
        start_time = pos * t_unit
        end_time = start_time + 0.5
        db_notes.append(
            pretty_midi.Note(start=start_time, end=end_time, pitch=DOWN_BEAT_NOTE_NUM, velocity=100)
        )

    beat_inst = pretty_midi.Instrument(is_drum=True, program=0, name="Beat")
    beat_inst.notes += beat_notes
    db_inst = pretty_midi.Instrument(is_drum=True, program=0, name="Down Beat")
    db_inst.notes += db_notes

    midi = pretty_midi.PrettyMIDI()
    midi.instruments = [beat_inst, db_inst]
    return midi
