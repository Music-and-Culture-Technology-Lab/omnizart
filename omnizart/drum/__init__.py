"""Drum transcription module.

Contains utilities for transcribe drum information in the music.
"""
import os

from omnizart.feature.cqt import extract_cqt
from omnizart.feature.beat_for_drum import extract_mini_beat_from_audio_path
from omnizart.drum.patched_cqt import extract_patch_cqt


def transcribe(input_audio, model_path, output="./"):
    """Transcribe drum in the audio.

    This function transcribes drum activations in the music. Currently the model
    predicts 13 classes of different drum sets, and 3 of them will be written to
    the MIDI file.

    Parameters
    ----------
    input_audio : Path
        Path to the raw audio file (.wav).
    model_path : Path
        Path to the trained model.
    output : Path (optional)
        Path for writing out the transcribed MIDI file. Default to current path.
    """
    if not os.path.isfile(input_audio):
        raise FileNotFoundError(f"The given audio path does not exist. Path: {input_audio}")

    # Load model configurations
    # ....

    # Extract feature according to model configuration
    cqt_feature = extract_cqt(input_audio)
    mini_beat_arr = extract_mini_beat_from_audio_path(input_audio)
    patch_cqt_feature = extract_patch_cqt(cqt_feature, mini_beat_arr)
