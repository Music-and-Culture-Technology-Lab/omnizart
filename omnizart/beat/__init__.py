"""MIDI domain beat tracking.

Track beats and downbeat in symbolic domain. Outputs the predicted beat positions
in seconds. Re-implementation of the work [1]_ with tensorflow 2.3.0.


Feature Storage Format
----------------------
Processed feature will be stored in ``.hdf`` format, one file per piece.

Columns in the file are:

* **feature**: Piano roll like representation with mixed information.
* **label**:


References
##########
.. [1] https://github.com/chuang76/symbolic-beat-tracking

"""

from omnizart.beat.app import BeatTranscription


app = BeatTranscription()
