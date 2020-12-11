"""Vocal pitch contour transcription.

Transcribes monophonic pitch contour of vocal in given polyphonic audio.
Re-implementation of the repository `Vocal-Melody-Extraction <https://github.com/s603122001/Vocal-Melody-Extraction>`_.

Feature Storage Format
----------------------
Processed feature and label will be stored in ``.hdf`` format, one file per piece.

Columns in the file are:

* **feature**: CFP feature representation.
* **label**: 2D numpy array of vocal pitch contour.

References
##########

The related publication of this work can be found in [1]_.

.. [1] Wei-Tsung Lu and Li Su, “Vocal melody extraction with semantic segmentation
   and audio-symbolic domain transfer learning,” International Society of Music
   Information Retrieval Conference (ISMIR), 2018.

"""

from omnizart.vocal_contour.app import VocalContourTranscription


app = VocalContourTranscription()
