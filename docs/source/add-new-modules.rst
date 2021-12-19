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
  Since the label extraction in this example is relatively simple, it is okay to put it under ``omnizart.patch_cnn.app``.
  See the conventional case :class:`omnizart.drum.labels`.

* `omnizart.patch_cnn._parallel_feature_extraction <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L336-L373>`_

  - To boost the process of feature extraction, files are processed in parallel. You can use the function
  :class:`omnizart.utils.parallel_generator` to accelerate the process.

* `omnizart.setting_loaders.PatchCNNSettings <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/setting_loaders.py#L330-L357>`_
  - The data class that holds the necessary hyper parameters that will be used by different functions of this module. For feature extraction, the
  parameters are registered under ``PatchCNNSettings.feature`` attribute.

* `omnizart/defatuls/patch_cnn.yaml <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/checkpoints/patch_cnn/patch_cnn_melody/configurations.yaml#L13-L53>`_
  - The configuration file of the module, records the values of hyper parameters and will be consumed by the data class (i.e. PatchCNNSettings).

Overall Process Flow
********************

1. Determine the dataset type from the given dataset path.
    * `music module <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/omnizart/music/app.py#L169-L179>`_
2. Choose the corresponding dataset structure class.
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L135>`_
    * `music module <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/omnizart/music/app.py#L182-L186>`_
3. Parse audio/ground-truth file pairs.
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L163-L167>`_
4. Make sure feature output path exists.
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L169-L172>`_
5. Parallel generate feature and label representation.
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L174-L188>`_
6. Write the settings to the output path, named as *.success.yaml*.
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L190-L193>`_


Implement Model Training
########################

All the training should happen in the ``.fit()`` function to fine-tune the model. There is supposed no need to manually
write the training loop.

Commits
*******

* `2d6f74d <https://github.com/Music-and-Culture-Technology-Lab/omnizart/pull/11/commits/2d6f74da88e52cef7ef6e96f3b93be97771bdf31>`_

Critical Files/Functions
************************

* `omnizart.patch_cnn.app.PatchCNNTranscription.train <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L195-L275>`_
  - The main function for managing the training flow.

* `omnizart.models.patch_cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/omnizart/models/patch_cnn.py>`_
  - Definition of the model. You can also customize the ``train_step`` function to do more sophisticated loss computation. See examples
  in `vocal <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/omnizart/models/pyramid_net.py#L233-L284>`_
  and `chord <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/omnizart/models/chord_model.py#L547-L600>`_
  modules.

* `omnizart.patch_cnn.app.PatchCNNDatasetLoader <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L376-L380>`_
  - The dataset loader for feeding data to models. Dealing with listing files, iterating through all feature/label pairs,
  indexing, and additionally augmenting, clipping, or transforming the feature/label on the fly.

* `omnizart.setting_loaders.PatchCNNSettings <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/setting_loaders.py#L366-L386>`_
  - The data class that holds the necessary hyper parameters that will be used by different functions of this module. For model training,
  related hyper parameters are registered under ``PatchCNNSettings.dataset``, ``PatchCNNSettings.model``, and
  ``PatchCNNSettings.training`` attributes.

* `omnizart/defatuls/patch_cnn.yaml (1) <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/checkpoints/patch_cnn/patch_cnn_melody/configurations.yaml#L54-L75>`_ /
  `omnizart/defatuls/patch_cnn.yaml (2) <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/checkpoints/patch_cnn/patch_cnn_melody/configurations.yaml#L88-L118>`_
  - The configuration file of the module, records the values of hyper parameters and will be consumed by the data class (i.e. PatchCNNSettings).

Overall Process Flow
********************

1. Check whether there is an input model or not. If given input model path, this indicating the user wants to fine-tune on a previously trained model. The coressponding settings should also be updated.
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L216-L219>`_
    * `drum <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/omnizart/drum/app.py#L167-L172>`_
2. Decide the portion of training and validation set.
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L221-L223>`_
    * `drum <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/omnizart/drum/app.py#L174-L176>`_
3. Construct dataset loader instances for training and validation.
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L225-L236>`_
4. Construct a fresh model if there is no input model.
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L238-L240>`_
5. Compile the model with loss function
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L242-L244>`_
    * `music <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/omnizart/drum/app.py#L167-L172>`_
6. Resolve the output path of the model
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L246-L255>`_
7. Construct the callbacks for storing the checkpoints, early stopping the training, and others.
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L257-L262>`_
8. Start training
    * `patch-cnn <https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/273fc60fbc6e3728c07abf71e06cf8f092bfabeb/omnizart/patch_cnn/app.py#L264-L274>`_


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

(To be continued...)
