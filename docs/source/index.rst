.. omnizart documentation master file, created by
   sphinx-quickstart on Tue Aug 25 10:43:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


OMNIZART: MUSIC TRANSCRIPTION MADE EASY
=======================================

.. figure:: ../../figures/features2.png
   :align: center


Omnizart is a Python library and a streamlined solution for automatic music transcription.
This library gathers the research outcomes from `Music and Cultural Technology Lab <https://sites.google.com/view/mctl/home>`_, 
analyzing polyphonic music and transcribes 
**musical notes of instruments** :cite:`music`,
**chord progression** :cite:`chord`,
**frame-level vocal melody** :cite:`vocalcontour`,
**note-level vocal melody**  :cite:`vocal`, and
**beat** :cite:`beat`.

Omnizart provides the main functionalities that construct the life-cycle of deep learning-based music transcription,
covering from *dataset downloading*, *feature pre-processing*, *model training*, to *transcription* and *sonification*.
Pre-trained checkpoints are also provided for the immediate usage of transcription.


Demo
####

Play with the `Colab notebook <https://bit.ly/omnizart-colab>`_ to transcribe your favorite song almost immediately!

Below is a demonstration of chord and drum transcription.

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/hjJhweRlE-A" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The result of chord transcription

.. raw:: html

   <audio controls="controls">
      <source src="_audio/high_chord_synth.mp3" type="audio/mpeg">
      Your browser does not support the <code>audio</code> element.
   </audio>


The result of drum transcription

.. raw:: html

   <audio controls="controls">
      <source src="_audio/high_drum_synth.mp3" type="audio/mpeg">
      Your browser does not support the <code>audio</code> element.
   </audio>



.. toctree::
   :maxdepth: 2
   :caption: Contents

   quick-start.rst
   tutorial.rst
   music/cli.rst
   drum/cli.rst
   chord/cli.rst
   vocal-contour/cli.rst


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   music/api.rst
   drum/api.rst
   chord/api.rst
   vocal-contour/api.rst
   feature.rst
   models.rst
   training.rst
   base.rst
   constants.rst
   utils.rst

.. Indices and tables
  ==================
   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

.. bibliography::
   refs.bib