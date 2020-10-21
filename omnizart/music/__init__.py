"""Music transcription module.

This module provides utilities for transcribing pitch and instruments in the audio.

Feature Storage Format
----------------------
Processed feature will be stored in ``.hdf`` and ``.pickle`` file format. The former format
is used to store the feature representation, and the later is used for customized label
representation. Each piece will have both two different files.

Columns in ``.hdf`` feature file:

* **feature**


References
##########
Technical details can be found in the publications [1]_ and [2]_.

.. [1] Wu, Yu-Te, Berlin Chen, and Li Su. "Automatic Music Yranscription Leveraging Generalized Cepstral Features and
   Deep Learning." IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.

.. [2] Wu, Yu-Te, Berlin Chen, and Li Su. "Polyphonic Music Transcription with Semantic Segmentation."
   IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019.
"""

from omnizart.music.app import MusicTranscription


app = MusicTranscription()
