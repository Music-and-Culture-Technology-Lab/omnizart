"""Vocal melody transcription.

Transcribe vocal notes in the song and output to MIDI file.
Re-implementation of the work [1]_ into tensorflow 2.3.

References
----------

.. [1] https://github.com/B05901022/VOCANO

"""
from omnizart.vocal.app import VocalTranscription


app = VocalTranscription()
