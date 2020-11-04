.. Documents are written in reStructured Text (.rst) format.
   Learn the syntax from: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
   
   Heading Level (most significant to least):
     Underline with '='
     Underline with '#'
     Underline with '*'


Tutorial
========

This page describes the basic concept and usage of ``omnizart`` command. 

The root entry is ``omnizart``, and followed by several sub-commands. To see a list of available sub-commands, type ``omnizart --help``.
All the sub-command follows the same structure (e.g. all have the ``transcribe`` command).

More detailed descriptions and the usage of each sub-commands can be found from their own page.


Transcribe
##########

Available sub-commands are:

* ``music`` - Trancribes instrument notes, outputs MIDI.
* ``drum`` - Transcribes drum percussions, outputs MIDI.
* ``chord`` - Transcribes chord progression, outputs MIDI and CSV.
* ``vocal`` *(preparing)* - Transcribes vocal melodies.
* ``beat`` *(preparing)* - MIDI-domain beat tracking.

Input for the former four should be wav file, and for ``beat`` module
should be a MIDI file.

Example Usage:

.. code-block:: bash

   # Use the default settings and the model
   omnizart music transcribe example.wav

   # Specify your own model and output path
   omnizart drum transcribe pop.wav --model-path ./my-model --output ./trans_pop.mid


Generate Feature
################

Generate the training feature of different datasets. Training and testing feature will be
stored in *<path/to/dataset>/train_feature* and *<path/to/dataset>/test_feature*, respectively.

Different module supports a subset of downloadable datasets. Datasets that each module supports
are listed below:

+-----------+-------+------+-------+------+-------+
| Module    | music | drum | chord | beat | vocal |
+===========+=======+======+=======+======+=======+
| Maestro   |   O   |      |       |      |       |
+-----------+-------+------+-------+------+-------+
| Maps      |   O   |      |       |      |       |
+-----------+-------+------+-------+------+-------+
| MusicNet  |   O   |      |       |      |       |
+-----------+-------+------+-------+------+-------+
| Pop       |   O   |  O   |       |      |       |
+-----------+-------+------+-------+------+-------+
| Ext-Su    |   O   |      |       |      |       |
+-----------+-------+------+-------+------+-------+
| BillBoard |       |      |   O   |      |       |
+-----------+-------+------+-------+------+-------+
| BPS-FH    |       |      |       |      |       |
+-----------+-------+------+-------+------+-------+


Example command for generating the feature is as following:

.. code-block:: bash

   omnizart music generate-feature --dataset-path <path/to/dataset>
   omnizart drum generate-feature --dataset-path <path/to/dataset>


Train Model
###########

After feature extraction finished, you can now train your own model~

.. code-block:: bash

   omnizart music train-model -d <path/to/feature/folder> --model-name My-Music
   omnizart drum train-model -d <path/to/feature/folder> --model-name My-Drum
   omnizart chord train-model -d <path/to/feature/folder> --model-name My-Chord


Download Datasets
#################

Download datasets for training models and evaluation by executing
``omnizart download-dataset <DATASET>``. Current supported datasets are:

* ``Maestro`` - MIDI and Audio Edited for Synchronous TRacks and Organization dataset.
* ``MusicNet`` - MusicNet dataset with a collection of 330 freely-licensed classical music recordings.
* ``McGill`` - McGill BillBoard dataset.
* ``BPS-FH`` - Beethoven Piano Sonata with Function Harmony dataset.
* ``Ext-Su`` - Extended Su dataset.

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

Download the archived checkpoints of different pre-trained models.
This command will download the checkpoints to where ``omnizart`` being installed.

.. code-block:: bash

   # Simply run the following command, and no other options need to be specified.
   omnizart download-checkpoints
