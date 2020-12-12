"""Vocal melody transcription.

Transcribe vocal notes in the song and outputs the MIDI file.
Re-implementation of the work [1]_ with tensorflow 2.3.0.
Some changes have also been made to improve the performance.

Feature Storage Format
----------------------
Processed feature will be stored in ``.hdf`` file format, one file per piece.

Columns in the file are:

* **feature**: CFP feature specialized for ``vocal`` module.
* **label**: Onset, offset, and duration information of the vocal.


References
##########

.. [1] https://github.com/B05901022/VOCANO


See Also
########

``omnizart.feature.cfp.extract_vocal_cfp``:
    Function to extract specialized CFP feature for ``vocal``.

"""
from omnizart.vocal.app import VocalTranscription


app = VocalTranscription()
