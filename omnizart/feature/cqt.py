import librosa
import numpy as np

from omnizart.io import load_audio
from omnizart.utils import get_logger


logger = get_logger("CQT Feature")


def post_process_cqt(gram):
    """
    Normalize and log-scale a Constant-Q spectrogram

    Parameters
    ----------
    gram: np.ndarray
        Constant-Q spectrogram, constructed from ``librosa.cqt``.

    Returns
    -------
    log_normalized_gram: np.ndarray
        Log-magnitude, L2-normalized constant-Q spectrogram.
    """
    # Compute log amplitude
    gram = (librosa.amplitude_to_db(np.abs(gram), amin=1e-06, top_db=80.0) + 80.001) * (100.0/80.0)  # noqa: E226

    # and L2 normalize
    gram = librosa.util.normalize(gram.T, norm=2.0, axis=1)
    return gram.astype(np.float32)


def extract_cqt(
    audio_path,
    sampling_rate=44100,
    lowest_note=16,
    note_num=120,
    a_hop=256,
    pad_sec=1
):
    """
    Compute some audio data's constant-Q spectrogram, normalize, and log-scale
    it

    Parameters
    ----------
    audio_data: Path
        Path to the input audio.
    sampling_rate: int
        Sampling rate the audio data is sampled at, should be ``DOWN_SAMPLE_TO_SAPMLING_RATE``.
    lowest_note: int
        Lowest MIDI note number.
    note_num: int
        Number of total notes. The highest note number would thus be `lowest_note` + `note_num`.
    a_hop: int
        Hop size for computing CQT.
    pad_sec: float
        Length of padding to the begin and the end of the raw audio data in seconds.

    Returns
    -------
    midi_gram: np.ndarray
        Log-magnitude, L2-normalized constant-Q spectrogram of synthesized MIDI
        data.
    """
    logger.debug("Loading audio: %s", audio_path)
    audio_data, _ = load_audio(audio_path, sampling_rate=sampling_rate)

    zeros = np.zeros(pad_sec * sampling_rate)
    padded_audio = np.concatenate([zeros, audio_data, zeros])

    # Compute CQT of the synthesized audio data
    logger.debug("Extracting CQT feature with librosa")
    audio_gram = librosa.cqt(
        padded_audio, sr=sampling_rate, hop_length=a_hop, fmin=librosa.midi_to_hz(lowest_note), n_bins=note_num
    )

    # L2-normalize and log-magnitute it
    logger.debug("Post-processing CQT feature...")
    return post_process_cqt(audio_gram)
