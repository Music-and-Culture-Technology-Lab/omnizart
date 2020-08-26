"""Music transcription module.

This module provides utilities for transcribing pitch and instruments in the audio.

References
----------
Technical details can be found in the publications [1]_ and [2]_.

.. [1] Wu, Yu-Te, Berlin Chen, and Li Su. "Automatic music transcription leveraging generalized cepstral features and
   deep learning." IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.

.. [2] Wu, Yu-Te, Berlin Chen, and Li Su. "Polyphonic music transcription with semantic segmentation." 
   IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019.
"""

import os

import numpy as np

from omnizart.music.model_manager import ModelManager
from omnizart.feature.cfp import extract_cfp
from omnizart.feature.hcfp import extract_hcfp
from omnizart.music.inference import multi_inst_note_inference
from omnizart.constants.feature import HOP
from omnizart.constants.midi import MUSICNET_INSTRUMENT_PROGRAMS


def transcribe(input_audio, model_path, output="./"):
    """ Transcribe notes and instruments of the given audio.

    This function transcribes notes (onset, duration) of each instruments in the audio.
    The results will be written out as a MIDI file.

    Parameters
    ----------
    input_audio : Path
        Path to the wav audio file.
    model_path : Path
        Path to the trained model. Should be the folder that contains `arch.yaml`, `weights.h5`, and `configuration.csv`.
    output : Path (optional)
        Path for writing out the transcribed MIDI file. Default to current path.

    See Also
    --------
    omnizart.cli.music.transcribe: The coressponding command line entry.
    """
    if not os.path.isfile(input_audio):
        raise FileNotFoundError(f"The given audio path does not exist. Path: {input_audio}")

    m_manage = ModelManager()
    model = m_manage.load_model(model_path)
    print(m_manage)

    # TODO: Add feature-related settings to the configuration.csv and load it in ModelInfo
    print("Extracting feature...")
    if m_manage.feature_type == "HCFP":
        spec, gcos, ceps, cenf = extract_hcfp(input_audio)
        feature = np.dstack([spec, gcos, ceps])
    else:
        z, spec, gcos, ceps, cenf = extract_cfp(input_audio)
        feature = np.dstack([z.T, spec.T, gcos.T, ceps.T])

    mode_mapping = {
        "frame": "true_frame",
        "frame_onset": "note",
        "multi_instrument_frame": "true_frame",
        "multi_instrument_note": "note-stream",
    }

    print("Predicting...")
    pred = m_manage.predict(feature[:, :, m_manage.input_channels], model)

    print("Infering notes....")
    midi = multi_inst_note_inference(
        pred,
        mode=mode_mapping[m_manage.label_type],
        onset_th=m_manage.onset_th,
        dura_th=m_manage.dura_th,
        frm_th=m_manage.frm_th,
        inst_th=m_manage.inst_th,
        t_unit=HOP,
        channel_program_mapping=MUSICNET_INSTRUMENT_PROGRAMS,
    )

    save_to = os.path.join(output, os.path.basename(input_audio).replace(".wav", ".mid"))
    midi.write(save_to)
    print(f"MIDI file has been written to {save_to}")
