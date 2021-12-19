Drum Transcription
==================


.. automodule:: omnizart.drum


App
###
.. autoclass:: omnizart.drum.app.DrumTranscription
    :members:
    :show-inheritance:


Dataset
#######
.. autoclass:: omnizart.drum.app.PopDatasetLoader
    :members:
    :show-inheritance:


Inference
#########
.. automodule:: omnizart.drum.inference
    :members:
    :undoc-members:


Labels
######
.. automodule:: omnizart.drum.labels
    :members:
    :undoc-members:


Prediction
##########
.. automodule:: omnizart.drum.prediction
    :members:
    :undoc-members:


Settings
########
Below are the default settings for building the drum model. It will be loaded
by the class :class:`omnizart.setting_loaders.DrumSettings`. The name of the
attributes will be converted to snake-case (e.g., HopSize -> hop_size). There
is also a path transformation process when applying the settings into the
``DrumSettings`` instance. For example, if you want to access the attribute
``BatchSize`` defined in the yaml path *General/Training/Settings/BatchSize*,
the corresponding attribute will be *DrumSettings.training.batch_size*.
The level of */Settings* is removed among all fields.

.. literalinclude:: ../../../omnizart/defaults/drum.yaml
    :language: yaml
