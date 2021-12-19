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

Core
####

In general, the core sub-commands follow a pipeline of ``application``-``action``-``arguments``:

.. code-block:: bash

   omnizart application action --arguments

where we apply an ``action`` to the ``application`` of interest, with corresponding ``arguments``.
Detailed descriptions for the usage of each sub-command can be found in the dedicated pages for each ``application``:

* :doc:`music/cli`
* :doc:`drum/cli`
* :doc:`chord/cli`
* :doc:`vocal-contour/cli`
* :doc:`vocal/cli`
* :doc:`beat/cli`

All the applications share a same set of actions: **transcribe**, **generate-feature**, and **train-model**.
Let's have a walkthrough of each ``action``.

Transcribe
**********

As the name suggests, this action transcribes a given input.
The supported applications are as follows:

* ``music`` - Transcribe musical notes of pitched instruments in MIDI.
* ``drum`` - Transcribe events of percussive instruments in MIDI.
* ``chord`` - Transcribe chord progressions in MIDI and CSV.
* ``vocal`` - Transcribe note-level vocal melody in MIDI.
* ``vocal-contour`` - Transcribe frame-level vocal melody (F0) in text.
* ``beat`` - Transcribe beat position.

Note that all the applications receive polyphonic music in WAV, except ``beat`` receives inputs in MIDI.

Example usage:

.. code-block:: bash

   # Transcribe percussive events given pop.wav, with specified model path and output directory
   omnizart drum transcribe pop.wav --model-path ./my-model --output ./trans_pop.mid

Note: ``--model-path`` can be left unspecified, and the default will be the downloaded checkpoints.
Execute ``omnizart download-checkpoints`` if you have not done in the installation from :doc:`quick-start`.


Generate Feature
****************

This action generates the features that are necessary for training and testing.
You can definitely skip this if you are only into transcribing with the given checkpoints.
The processed features will be stored in *<path/to/dataset>/train_feature* and *<path/to/dataset>/test_feature*.

The supported datasets for feature processing are application-dependent, summarized as follows:

+-------------+-------+------+-------+------+-------+---------------+------+
| Module      | music | drum | chord | beat | vocal | vocal-contour | beat |
+=============+=======+======+=======+======+=======+===============+======+
| Maestro     |   O   |      |       |      |       |               |      |
+-------------+-------+------+-------+------+-------+---------------+------+
| Maps        |   O   |      |       |      |       |               |      |
+-------------+-------+------+-------+------+-------+---------------+------+
| MusicNet    |   O   |      |       |      |       |               |  O   |
+-------------+-------+------+-------+------+-------+---------------+------+
| Pop         |   O   |  O   |       |      |       |               |      |
+-------------+-------+------+-------+------+-------+---------------+------+
| Ext-Su      |   O   |      |       |      |       |               |      |
+-------------+-------+------+-------+------+-------+---------------+------+
| BillBoard   |       |      |   O   |      |       |               |      |
+-------------+-------+------+-------+------+-------+---------------+------+
| BPS-FH      |       |      |       |      |       |               |      |
+-------------+-------+------+-------+------+-------+---------------+------+
| MIR-1K      |       |      |       |      |   O   |       O       |      |
+-------------+-------+------+-------+------+-------+---------------+------+
| MedleyDB    |       |      |       |      |       |       O       |      |
+-------------+-------+------+-------+------+-------+---------------+------+
| Tonas       |       |      |       |      |   O   |               |      |
+-------------+-------+------+-------+------+-------+---------------+------+

Before running the commands below, make sure to download the corresponding datasets first.
This can be easily done in :ref:`Download Datasets`.

.. code-block:: bash

   # Generate features for the music application
   omnizart music generate-feature --dataset-path <path/to/dataset>

   # Generate features for the drum application
   omnizart drum generate-feature --dataset-path <path/to/dataset>


Train Model
***********

This action trains a model from scratch given the generated features from :ref:`Generate Feature`.
Once again, you can skip this if you are only up to transcribing music, and use the provided checkpoints.

.. code-block:: bash

   omnizart music train-model -d <path/to/feature/folder> --model-name My-Music
   omnizart drum train-model -d <path/to/feature/folder> --model-name My-Drum
   omnizart chord train-model -d <path/to/feature/folder> --model-name My-Chord


Utility
#######


Download Datasets
*****************

This sub-command belongs to the utility, used to download the datasets for training and testing the models.
Current supported datasets are:

* `Maestro <https://magenta.tensorflow.org/datasets/maestro>`_ - MIDI and Audio Edited for Synchronous TRacks and Organization dataset.
* `MusicNet <https://homes.cs.washington.edu/~thickstn/musicnet.html>`_ - MusicNet dataset with a collection of 330 freely-licensed classical music recordings.
* `McGill <https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/>`_ - McGill BillBoard dataset.
* `BPS-FH <https://github.com/Tsung-Ping/functional-harmony>`_ - Beethoven Piano Sonata with Function Harmony dataset.
* Ext-Su - Extended Su dataset.
* `MIR-1K <https://sites.google.com/site/unvoicedsoundseparation/mir-1k>`_ - 1000 short clips of Mandarin pop songs.
* `MedleyDB <http://medleydb.weebly.com/>`_ - 122 multitracks.

Example usage:

.. code-block:: bash

   # Download the MAESTRO dataset and output to the */data* folder.
   omnizart download-dataset Maestro --output /data

   # Download the MusicNet dataset and unzip the dataset after download.
   omnizart download-dataset MusicNet --unzip

   # To see a complete list of available datasets, execute the following command
   omnizart download-dataset --help


Download Checkpoints
********************

This is the other sub-command for the utility, used to download the archived checkpoints of pre-trained models.

.. code-block:: bash

   # Simply run the following command, and no other options are needed to be specified.
   omnizart download-checkpoints
