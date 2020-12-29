Add New Modules
===============

This page describes how to add new modules into Omnizart project, adapt the original implementations
to omnizart's architecture.

Before starting walking through the integration process, be sure you have already read the
`CONTRIBUTING.md <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/CONTRIBUTING.md>`_,
and the `slides of omnziart <https://drive.google.com/file/d/1IO1lh07nMvSi0X0nzRDT7kuE1f468Rl1/view?usp=sharing>`_
could also be helpful for your understanding of this project.
Additionally, there are few more things to be always kept in mind while developing omnizart.

Principles
##########

* **Find an existing module and start development** - There are already several implemented modules
  that are fully functional, and being great examples that give you hints on your way developing
  new modules. Most of them are very similar of their overall architecture, but vary in detail.
  Most the time, you could just copy and paste the small pieces to your module, and modify just a
  small part of them to adapt to your task.
* **Try not to make your own wheels** - There have been many useful and validated functions that are
  developed to deal with the daily works. They are already there to cover 90% of every details of a
  module, thus new logics are in very small chances being needed. 
  Most of the time you need to implement the most would be the part of feature and label extraction,
  which will be explained in the upcoming sections.
* **Check with linters frequently** - You should always do ``make lint`` before you push to github,
  checking that there aren't any errors with the code format, or the build process would fail.
* **Don't permit linter errors easily** - You may find some comments that permits the linter errors
  while surfing the code. Those are quick solutions while in the early development of omnizart, which
  saves lots of time fixing those lint errors. But it should not be the main concern now, as the
  architecture is more stable and less error prone. You should follow every hints by the linters
  and fix them before you file a pull request.
 

----

So now we are all set and ready to add a new module to omnizart. Here we will take the 
`PR #11 <https://github.com/Music-and-Culture-Technology-Lab/omnizart/pull/11>`_ as the example.

Setup
#####

1. **IMPORTANT** - Give your module a short, yet descriptive name. In the example, the name is
   ``PatchCNN`` (camel-case), ``patch_cnn`` (snake-case), and ``patch-cnn`` (for CLI use).

2. Create a folder named after your module under ``omnizart``. There should be at least two files:
   ``app.py`` and ``__init__.py``.

Implement Feature Generation
############################

The process unit should be **a dataset**, means the function accepts the path to the dataset itself, and will handle the rest
of the things like dataset type inferring, folder structure handling, file parsing, feature extraction, and output storage.

Commits
*******

* `3ff6c4a <https://github.com/Music-and-Culture-Technology-Lab/omnizart/pull/11/commits/3ff6c4abe5ab98242d33c146353b5282ce5f6b66>`_
  - Builds the main structure of feature extraction.
* `f3138eb <https://github.com/Music-and-Culture-Technology-Lab/omnizart/pull/11/commits/f3138eb4a0650c91692f70e09bab1578be11c132>`_
  - Contains the patch of label extraction function.
* `0190f18 <https://github.com/Music-and-Culture-Technology-Lab/omnizart/pull/11/commits/0190f1895027cf859647c2099d3c03a24f73246a>`_
  - Contains the patch of label extraction function.

Critical Files/Functions
************************

* `omnizart.patch_cnn.app.PatchCNNTranscription.generate_feature <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L106-L193>`_
  - The main function for managing the process of feature generation.

* `omnizart.feature.cfp.extract_patch_cfp <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/feature/cfp.py#L355-L451>`_
  - The function for feature extraction, which takes audio path as the input and outputs the required feature representations.

* `omnizart.patch_cnn.app.extract_label <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L278-L327>`_
  - The function for label extraction, generating the representation of ground-truth. Accepts the path to the ground-truth file, parses the contents
  into intermediate format (see :class:`omnizart.base.Label`), and extracts necessary informations.
  
  Normally, it should be defined in a separate file called ``labels.py`` under *omnizart/<module>/* when the extraction process contains lots of logics.
  Since the label extraction in this example is relativly simple, it is okay to put it under ``omnizart.patch_cnn.app``.
  See the conventional case :class:`omnizart.drum.labels`.

* `omnizart.patch_cnn._parallel_feature_extraction <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L336-L373>`_
  - To boost the process of feature extraction, files are processed in parallel. You can use the function :class:`omnizart.utils.parallel_generator`
  to accelerate the process.

* `omnizart.setting_loaders.PatchCNNSettings <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/setting_loaders.py#L330-L357>`_
  - The data class that holds the necessary hyper parameters that will be used by different functions of this module. For feature extraction, the
  parameters are registered under ``PatchCNNSettings.feature`` attribute.

* `omnizart/defatuls/patch_cnn.yaml <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/checkpoints/patch_cnn/patch_cnn_melody/configurations.yaml#L13-L53>`_
  - The configuration file of the module, records the values of hyper parameters and will be consumed by the data class (i.e. PatchCNNSettings).

Overall Process Flow
********************

1. Determine the dataset type from the given dataset path.
    * `music module <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/omnizart/music/app.py#L169-L179>`_
2. Choose the coressponding dataset strcuture class.
    * `patch-cnn module <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L135>`_
    * `music module <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/omnizart/music/app.py#L182-L186>`_
3. Parse audio/ground-truth file pairs.
    * `patch-cnn module <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L163-L167>`_
4. Make sure feature output path exists.
    * `patch-cnn module <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L169-L172>`_
5. Parallel generate feature and label representation.
    * `patch-cnn module <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L174-L188>`_
6. Write the settings to the output path, named as *.success.yaml*.
    * `patch-cnn module <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L190-L193>`_

Implement Model Training
########################

Implement Transcription
#######################

Add Unit Tests
##############

Commit Checkpoints
##################

Implement CLI
#############

Add Documentation
#################

----

Optional
########

This section holds the optional actions you can do, while it is not necessary to be done
during implementing a new module.

Add new supported datasets
**************************

If you want to add a new dataset that is currently not supported by ``omnizart`` (which is defined in
:class:`omnizart.constants.datasets`), things should be noticed are explained in this section.

(To be continue...)

