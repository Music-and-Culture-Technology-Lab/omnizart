Patch-CNN Transcription
=======================


.. automodule:: omnizart.patch_cnn


App
###
.. autoclass:: omnizart.patch_cnn.app.PatchCNNTranscription
    :members:
    :show-inheritance:


Dataset
#######
.. autoclass:: omnizart.patch_cnn.app.PatchCNNDatasetLoader
    :members:
    :show-inheritance:


Inference
#########
.. automodule:: omnizart.patch_cnn.inference
    :members:


Labels
######
.. autofunction:: omnizart.patch_cnn.app.extract_label



Settings
########
Below are the default settings for building the PatchCNN model. It will be loaded
by the class :class:`omnizart.setting_loaders.PatchCNNSettings`. The name of the
attributes will be converted to snake-case (e.g. HopSize -> hop_size). There
is also a path transformation process when applying the settings into the
``PatchCNNSettings`` instance. For example, if you want to access the attribute
``BatchSize`` defined in the yaml path *General/Training/Settings/BatchSize*,
the coressponding attribute will be *MusicSettings.training.batch_size*.
The level of */Settings* is removed among all fields.

.. literalinclude:: ../../../omnizart/defaults/patch_cnn.yaml
    :language: yaml
