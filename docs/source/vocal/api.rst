Vocal Transcription
===================


.. automodule:: omnizart.vocal


App
###
.. autoclass:: omnizart.vocal.app.VocalTranscription
    :members:
    :show-inheritance:


Dataset
#######
.. autoclass:: omnizart.vocal.app.VocalDatasetLoader
    :members:
    :show-inheritance:


Inference
#########
.. automodule:: omnizart.vocal.inference
    :members:


Labels
######
.. automodule:: omnizart.vocal.labels
    :members:
    :undoc-members:


Prediction
##########
.. automodule:: omnizart.vocal.prediction
    :members:
    :undoc-members:


Settings
########
Below are the default settings for building the vocal model. It will be loaded
by the class :class:`omnizart.setting_loaders.VocalSettings`. The name of the
attributes will be converted to snake-case (e.g. HopSize -> hop_size). There
is also a path transformation process when applying the settings into the
``VocalSettings`` instance. For example, if you want to access the attribute
``BatchSize`` defined in the yaml path *General/Training/Settings/BatchSize*,
the coressponding attribute will be *VocalSettings.training.batch_size*.
The level of */Settings* is removed among all fields.

.. literalinclude:: ../../../omnizart/defaults/vocal.yaml
    :language: yaml
