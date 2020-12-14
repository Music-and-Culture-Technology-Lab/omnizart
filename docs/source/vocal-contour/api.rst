Vocal-Contour Transcription
===========================


.. automodule:: omnizart.vocal_contour


App
###
.. automodule:: omnizart.vocal_contour.app
    :members:
    :show-inheritance:


Inference
#########
.. automodule:: omnizart.vocal_contour.inference
    :members:


Loss Functions
##############
.. automodule:: omnizart.music.losses
    :members:


Settings
########
Below are the default settings for frame-level vocal transcription. 
It will be loaded by the class :class:`omnizart.setting_loaders.VocalContourSettings`. 
The name of the attributes will be converted to snake-case (e.g. HopSize -> hop_size). 
There is also a path transformation when applying the settings into the ``VocalContourSettings`` instance. 
For example, the attribute ``BatchSize`` defined in the yaml path *General/Training/Settings/BatchSize* is transformed 
to *VocalContourSettings.training.batch_size*. 
The level of */Settings* is removed among all fields.

.. literalinclude:: ../../../omnizart/defaults/vocal_contour.yaml
    :language: yaml
