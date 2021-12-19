Chord Transcription
===================


.. automodule:: omnizart.chord


App
###
.. autoclass:: omnizart.chord.app.ChordTranscription
    :members:
    :show-inheritance:


Feature
#######
.. automodule:: omnizart.chord.features
    :members:
    :undoc-members:


Dataset
#######
.. autoclass:: omnizart.chord.app.McGillDatasetLoader
    :members:
    :show-inheritance:


Inference
#########
.. automodule:: omnizart.chord.inference
    :members:
    :undoc-members:


Settings
########
Below are the default settings for building the chord model. It will be loaded
by the class :class:`omnizart.setting_loaders.ChordSettings`. The name of the
attributes will be converted to snake-case (e.g., HopSize -> hop_size). There
is also a path transformation process when applying the settings into the
``ChordSettings`` instance. For example, if you want to access the attribute
``BatchSize`` defined in the yaml path *General/Training/Settings/BatchSize*,
the corresponding attribute will be *ChordSettings.training.batch_size*.
The level of */Settings* is removed among all fields.

.. literalinclude:: ../../../omnizart/defaults/chord.yaml
    :language: yaml
