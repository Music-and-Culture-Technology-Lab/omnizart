import vamp

from omnizart.io import load_audio


AVAILABLE_OUTPUT_TYPES = ["logfreqspec", "tunedlogfreqspec", "semitonespectrum", "chroma", "basschroma", "bothchroma"]
TUNING_MODE = {"global": 0, "local": 1}
CHROMA_NORM = {"none": 0, "max": 1, "l1": 2, "l2": 3}


def extract_chroma(
    audio_path,
    output_type: str = "bothchroma",
    tuning_mode: str = "global",
    chroma_norm: str = "none",
    use_nnls: bool = False,
    roll_on: int = 1,
    spectral_whitening: float = 1,
    spectral_shape: float = 0.7
):
    """Chroma feature extraction with Vamp.

    This function extracts the chroma feature by using the library `Vamp <http://www.isophonics.net/nnls-chroma>`_,
    which is also used by the McGill Billboard dataset for generating the training feature.
    Detailed functions and corresponding parameters can be found in ``omnizart/resource/vamp/nnls-chroma.n3``.

    Parameters
    ----------
    audio_path: Path
        Path to the input audio.
    output_type: {"logfreqspec", "tunedlogfreqspec", "semitonespectrum", "chroma", "basschroma", "bothchroma"}
        Type of chroma feature to be extracted.
    use_nnls: bool
        Use approximate transcription (NNLS).
    roll_on: 0 <= int <= 5
        Extend of removing low-frequency noise (in %).
    tuning_mode: {"global", "local"}
        Compute average locally or globally.
    spectral_whitening: 0 <= float <= 1
        Determines how much the log-frequency spectrum is whitened.
    spectral_shape: 0.5 <= float <= 0.9
        The shape of the notes in the NNLS dictionary.
    chroma_norm: {"none", "max", "l1", "l2"}
        Determines whether or how the chromagrams are normalized.

    References
    ----------
    The python version of Vamp can be found in [1]_. Also please check out the official site [2]_
    for a more comprehensive explaination.

    .. [1] https://github.com/c4dm/vampy-host
    .. [2] http://www.isophonics.net/nnls-chroma
    """
    assert output_type in AVAILABLE_OUTPUT_TYPES, f"Invalid output type: {output_type}. \
        Available options: {AVAILABLE_OUTPUT_TYPES}"
    assert tuning_mode in TUNING_MODE, f"Invalid tuninig mode: {tuning_mode}. \
        Should be one of {TUNING_MODE}."
    assert chroma_norm in CHROMA_NORM, f"Invalid chroma normalization mode: {chroma_norm}. \
        Should be one of {CHROMA_NORM}."
    assert 0 <= roll_on <= 5 and isinstance(roll_on, int), f"Invalid range/type of roll on: {roll_on}. \
        Should be type of integer and value within 0~5."
    assert 0 <= spectral_whitening <= 1, f"Invalid range of whitening value: {spectral_whitening}. \
        Should have value within 0~1."
    assert 0.5 <= spectral_shape <= 0.9, f"Invalid range of spectral shape: {spectral_shape}. \
        Should have value within 0.5~0.9."

    params = {
        "useNNLS": 1 if use_nnls else 0,
        "rollon": roll_on,
        "tuningmode": TUNING_MODE[tuning_mode],
        "whitening": spectral_whitening,
        "s": spectral_shape,
        "chromanormalize": CHROMA_NORM[chroma_norm]
    }

    data, rate = load_audio(audio_path)
    step_size, chroma = vamp.collect(
        data, rate, "nnls-chroma:nnls-chroma", output=output_type, parameters=params
    )["matrix"]
    return step_size.to_float(), chroma
