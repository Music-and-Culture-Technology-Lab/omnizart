Beat Transcription
===================


.. automodule:: omnizart.beat


App
###
.. autoclass:: omnizart.beat.app.BeatTranscription
    :members:
    :show-inheritance:


Dataset
#######
.. autoclass:: omnizart.beat.app.BeatDatasetLoader
    :members:
    :show-inheritance:


Inference
#########
.. automodule:: omnizart.beat.inference
    :members:


Loss Functions
##############
.. autofunction:: omnizart.beat.app.weighted_binary_crossentropy


Features
########
.. automodule:: omnizart.beat.features
    :members:


Prediction
##########
.. automodule:: omnizart.beat.prediction
    :members:


Settings
########
Below are the default settings for building the beat model. It will be loaded
by the class :class:`omnizart.setting_loaders.BeatSettings`. The name of the
attributes will be converted to snake-case (e.g., HopSize -> hop_size). There
is also a path transformation process when applying the settings into the
``BeatSettings`` instance. For example, if you want to access the attribute
``BatchSize`` defined in the yaml path *General/Training/Settings/BatchSize*,
the corresponding attribute will be *BeatSettings.training.batch_size*.
The level of */Settings* is removed among all fields.

.. literalinclude:: ../../../omnizart/defaults/beat.yaml
    :language: yaml
