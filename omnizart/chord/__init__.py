"""Chord transcription for both MIDI and audio domain.

Re-implementation of the repository
`Tsung-Ping/Harmony-Transformer <https://github.com/Tsung-Ping/Harmony-Transformer>`_.

Feature Storage Format
----------------------
Processed feature will be stored in ``.hdf`` file format, one file per piece.

Columns in the file are:

* **chroma**: the input feature for audio domain data.
* **chord**: the first type of ground-truth label.
* **chord_change**: the second type of ground-truth label.
* **tc**
* **sequence_len**
* **num_sequence**

References
##########

The related publications and techinical details can be found in [1]_ and [2]_

.. [1] Tsung-Ping Chen and Li Su, "Harmony Transformer: Incorporating Chord
   Segmentation Into Harmony Recognition," International Society of
   Music Information Retrieval Conference (ISMIR), 2019.
.. [2] Tsung-Ping Chen and Li Su, “Functional Harmony Recognition with
   Multi-task Recurrent Neural Networks,” International Society of Music Information
   Retrieval Conference (ISMIR), September 2018
"""

from omnizart.chord.app import ChordTranscription


app = ChordTranscription()
