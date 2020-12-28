"""Vocal pitch contour transcription PatchCNN ver.

Transcribes monophonic pitch contour of vocal in the given polyphonic audio
by using the PatchCNN approach.
Re-implementation of the repository `VocalMelodyExtPatchCNN <https://github.com/leo-so/VocalMelodyExtPatchCNN>`_.

Feature Storage Format
----------------------
Processed feature and label will be stored in ``.hdf`` format, one file per piece.

Columns contained in each file are:

* **feature**: Patch CFP feature.
* **label**: Binary classes of each patch.
* **Z**: The original CFP feature.
* **mapping**: Records the original frequency and time indexes of each patch.

References
##########

Publication of this module can be found in [1]_.

.. [1] Li Su, "Vocal Melody Extraction Using Patch-based CNN," in IEEE International Conference of Acoustics,
   Speech, and Signal Processing (ICASSP), 2018.

"""

from omnizart.patch_cnn.app import PatchCNNTranscription


app = PatchCNNTranscription()
