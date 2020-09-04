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

from omnizart.music.app import app


app = MusicTranscription()
