"""Drum transcription module.

Utilities for transcribing drum percussions in the music.

Feature Storage Format
----------------------
Processed feature will be stored in ``.hdf`` file format, one file per piece.

Columns in the file are:

* **feature**: CQT feature.
* **label**: Merged drum label set, with a total of 13 classes.
* **label_128**: Complete drum label set.
* **mini_beat_arr**: The tracked mini-beat array of the clip.

Example
#######
>>> import h5py
>>> hdf_ref = h5py.File("ytd_audio_00001_TRBSAIC128E0793CCE.hdf", "r")
>>> hdf_ref.keys()
<KeysViewHDF5 ['mini_beat_arr', 'feature', 'label', 'label_128']>
>>> feature = hdf_ref["feature"][:]
>>> print(feature.shape)
(2299, 120, 120)
>>> hdf_ref.close()

References
##########

The relative publication can be found in [1]_

.. [1] I-Chieh Wei, Chih-Wei Wu, Li Su. "Improving Automatic Drum Transcription Using Large-Scale
   Audio-to-MIDI Aligned Data" (in submission)

"""

from omnizart.drum.app import DrumTranscription

app = DrumTranscription()
