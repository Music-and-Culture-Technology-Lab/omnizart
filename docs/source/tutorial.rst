.. Documents are written in reStructured Text (.rst) format.
   Learn the syntax from: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
   
   Heading Level (most significant to least):
     Underline with '='
     Underline with '#'
     Underline with '*'


Tutorial
========

This page describes the workflow and usage of ``omnizart`` command-line interface, 
covering core and utility.

The root entry is ``omnizart`` followed by sub-commands.
The available sub-commands can be found by typing ``omnizart --help``.

In general, the core sub-commands follow a pipeline of ``application``-``action``-``arguments``:

.. code-block:: bash

   omnizart application action --arguments

where we apply an ``action`` (among ``transcribe``, ``generate-feature``, and ``train-model``) to
the ``application`` of interest, with corresponding ``arguments``.
Detailed descriptions for the usage of each sub-command can be found in the dedicated pages for each application.


Transcribe
##########

This action transcribes a given input.
The sub-commands that follow are the applications:

* ``music`` - Transcribes polyphonic music, and outputs notes of pitched instruments in MIDI.
* ``drum`` - Transcribes polyphonic music, and outputs events of percussive instruments in MIDI.
* ``chord`` - Transcribes polyphonic music, and outputs chord progression in MIDI and CSV.
* ``vocal-contour`` - Transcribes polyphonic music, and outputs frame-level vocal melody (F0) in text.
* ``vocal`` *(preparing)* - Transcribes polyphonic music, and outputs note-level vocal melody.
* ``beat`` *(preparing)* - MIDI-domain beat tracking.

Except ``beat`` which takes as input a MIDI file, all the applications receive audio files in WAV.

Example usage:

.. code-block:: bash

   # Transcribe percussive events given pop.wav, with specified model path and output directory
   omnizart drum transcribe pop.wav --model-path ./my-model --output ./trans_pop.mid

Note that `--model-path` and `--output` can also be left unspecified, and the defaults will be applied.

Generate Feature
################

This action generates the features that are necessary for training and testing.
The features will be stored in *<path/to/dataset>/train_feature* and *<path/to/dataset>/test_feature*.

Different modules of applications support different datasets, as follows:

+-------------+-------+------+-------+------+-------+---------------+
| Module      | music | drum | chord | beat | vocal | vocal-contour |
+=============+=======+======+=======+======+=======+===============+
| Maestro     |   O   |      |       |      |       |               |
+-------------+-------+------+-------+------+-------+---------------+
| Maps        |   O   |      |       |      |       |               |
+-------------+-------+------+-------+------+-------+---------------+
| MusicNet    |   O   |      |       |      |       |               |
+-------------+-------+------+-------+------+-------+---------------+
| Pop         |   O   |  O   |       |      |       |               |
+-------------+-------+------+-------+------+-------+---------------+
| Ext-Su      |   O   |      |       |      |       |               |
+-------------+-------+------+-------+------+-------+---------------+
| BillBoard   |       |      |   O   |      |       |               |
+-------------+-------+------+-------+------+-------+---------------+
| BPS-FH      |       |      |       |      |       |               |
+-------------+-------+------+-------+------+-------+---------------+
| MIR-1K      |       |      |       |      |   O   |       O       |
+-------------+-------+------+-------+------+-------+---------------+
| MedleyDB    |       |      |       |      |       |       O       |
+-------------+-------+------+-------+------+-------+---------------+
| Tonas       |       |      |       |      |   O   |               |
+-------------+-------+------+-------+------+-------+---------------+


Example usage:

.. code-block:: bash

   # Generate features for the music application
   omnizart music generate-feature --dataset-path <path/to/dataset>

   # Generate features for the drum application
   omnizart drum generate-feature --dataset-path <path/to/dataset>


Train Model
###########

This action trains a model from scratch given the generated features.

.. code-block:: bash

   omnizart music train-model -d <path/to/feature/folder> --model-name My-Music
   omnizart drum train-model -d <path/to/feature/folder> --model-name My-Drum
   omnizart chord train-model -d <path/to/feature/folder> --model-name My-Chord


Download Datasets
#################

This sub-command belongs to the utility, used to download the datasets for 
training and testing the models. 
Current supported datasets are:

* ``Maestro`` - MIDI and Audio Edited for Synchronous TRacks and Organization dataset.
* ``MusicNet`` - MusicNet dataset with a collection of 330 freely-licensed classical music recordings.
* ``McGill`` - McGill BillBoard dataset.
* ``BPS-FH`` - Beethoven Piano Sonata with Function Harmony dataset.
* ``Ext-Su`` - Extended Su dataset.
* ``MIR-1K`` - 1000 clips of Mandarin pop songs, with background music and vocal recorded in separated channels.

Example usage:

.. code-block:: bash

   # Download the MAESTRO dataset and output to the */data* folder.
   omnizart download-dataset Maestro --output /data

   # Downlaod the MusicNet dataset and unzip the dataset after download.
   omnizart download-dataset MusicNet --unzip

   # To see a complete list of available datasets, execute the following command
   omnizart download-dataset --help


Download Checkpoints
####################

This is the other sub-command for the utility, used to download the archived checkpoints of pre-trained models.

.. code-block:: bash

   # Simply run the following command, and no other options are needed to be specified.
   omnizart download-checkpoints
