Music Transcription
===================


.. automodule:: omnizart.music


App
###
.. autoclass:: omnizart.music.app.MusicTranscription
    :members:
    :show-inheritance:


Dataset
#######
.. autoclass:: omnizart.music.app.MusicDatasetLoader
    :members:
    :show-inheritance:


Inference
#########
.. automodule:: omnizart.music.inference
    :members:


Loss Functions
##############
.. automodule:: omnizart.music.losses
    :members:


Labels
######
.. automodule:: omnizart.music.labels
    :members:
    :undoc-members:


Prediction
##########
.. automodule:: omnizart.music.prediction
    :members:


Settings
########
Below are the default settings for building the music model. It will be loaded
by the class :class:`omnizart.setting_loaders.MusicSettings`. The name of the
attributes will be converted to snake-case (e.g., HopSize -> hop_size). There
is also a path transformation process when applying the settings into the
``MusicSettings`` instance. For example, if you want to access the attribute
``BatchSize`` defined in the yaml path *General/Training/Settings/BatchSize*,
the corresponding attribute will be *MusicSettings.training.batch_size*.
The level of */Settings* is removed among all fields.

.. literalinclude:: ../../../omnizart/defaults/music.yaml
    :language: yaml
