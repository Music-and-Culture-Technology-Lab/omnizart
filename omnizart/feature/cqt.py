import librosa
import numpy as np

from omnizart.io_utils import load_audio_with_librosa
from omnizart.constants.feature import DOWN_SAMPLE_TO_SAPMLING_RATE, LOWEST_NOTE, NUMBER_OF_NOTES, HOP_SIZE, PAD_LEN_SEC


def post_process_cqt(gram):
    """
    Normalize and log-scale a Constant-Q spectrogram

    Parameters
    ----------
    gram : np.ndarray
        Constant-Q spectrogram, constructed from ``librosa.cqt``.

    Returns
    -------
    log_normalized_gram : np.ndarray
        Log-magnitude, L2-normalized constant-Q spectrogram.
    """
    # Compute log amplitude
    gram = (librosa.amplitude_to_db(np.abs(gram), amin=1e-06, top_db=80.0) + 80.001) * (100.0/80.0)  # noqa: E226

    # and L2 normalize
    gram = librosa.util.normalize(gram.T, norm=2.0, axis=1)
    return gram.astype(np.float32)


def extract_cqt(
    audio_path,
    sampling_rate=DOWN_SAMPLE_TO_SAPMLING_RATE,
    lowest_note=LOWEST_NOTE,
    note_num=NUMBER_OF_NOTES,
    a_hop=HOP_SIZE,
):
    """
    Compute some audio data's constant-Q spectrogram, normalize, and log-scale
    it

    Parameters
    ----------
    audio_data : Path
        Path to the input audio.
    sampling_rate : int
        Sampling rate the audio data is sampled at, should be ``DOWN_SAMPLE_TO_SAPMLING_RATE``.

    Returns
    -------
    midi_gram : np.ndarray
        Log-magnitude, L2-normalized constant-Q spectrogram of synthesized MIDI
        data.
    """
    audio_data, _ = load_audio_with_librosa(audio_path, sampling_rate=sampling_rate)

    zeros = np.zeros(PAD_LEN_SEC * sampling_rate)
    padded_audio = np.concatenate([zeros, audio_data, zeros])

    # Compute CQT of the synthesized audio data
    audio_gram = librosa.cqt(
        padded_audio, sr=sampling_rate, hop_length=a_hop, fmin=librosa.midi_to_hz(lowest_note), n_bins=note_num
    )

    # L2-normalize and log-magnitute it
    return post_process_cqt(audio_gram)
