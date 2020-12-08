"""Music transcription module.

This module provides utilities for transcribing pitch and instruments in the audio.
This module is also an improved version of the original repository
`BreezeWhite/Music-Transcription-with-Semantic-Segmentation
<https://github.com/BreezeWhite/Music-Transcription-with-Semantic-Segmentation>`_,
with a cleaner architecture and consistent coding style, also provides command line interface
for easy usage.

Feature Storage Format
----------------------
Processed feature will be stored in ``.hdf`` and ``.pickle`` file format. The former format
is used to store the feature representation, and the later is used for customized label
representation. Each piece will have both two different files.

Columns in ``.hdf`` feature file:

* **feature**


References
##########
Technical details can be found in the publications [1]_, [2]_, and [3]_.

.. [1] Yu-Te Wu, Berlin Chen, and Li Su, "Multi-Instrument Automatic Music Transcription With Self-Attention-Based
   Instance Segmentation." in IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2020.

.. [2] Yu-Te Wu, Berlin Chen, and Li Su. "Polyphonic Music Transcription with Semantic Segmentation."
   IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019.

.. [3] Yu-Te Wu, Berlin Chen, and Li Su. "Automatic Music Yranscription Leveraging Generalized Cepstral Features and
   Deep Learning." IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.

"""

from omnizart.music.app import MusicTranscription


app = MusicTranscription()
