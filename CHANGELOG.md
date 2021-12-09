# Changelog

## 0.5.0 - 2021-12-09

Official Open JOSS reviewed version.

## Bugs
- Fix bug of name conflict while loading chord model.


## 0.4.2 - 2021-11-16

Accumulated release. Various improvements and bug fix. See details below.

## Feature
- Migrate checkpoints from private Google Drive to Github release. 
See [here](https://github.com/Music-and-Culture-Technology-Lab/omnizart/releases/tag/checkpoints-20211001)
- Replace opencv 

## Dependency
- Upgrade Tensorflow version to 2.5.0 for Nvidia 30 series GPU compatibility.
- Upgrade Spleeter version to 2.3.0 for new TF version compatibility.
- Replace Opencv with PIL for drum feature resizing and remove opencv from the dependency.

## Enhancement
- Simplify model loading mechanism by unifying the all checkpoint format to use TF format.
- Lazy import extraction functions to boost loading time.
- Change the order of Dockerfile commands for better utilizing cache.

## Documentation
- Add notice about compatibility issue of running on certain CPU architecture.
- Add explaination about enabling auto completion.
- Rephrase sentences in paper according to JOSS review feedback.
- Add explaination about installing development dependencies.
- Use pepy as the alternative source for 'download' state badge.


## Bugs
- Fix bug of unable to find vocal contour checkpoint.
- Fix bug of fail to custom layers of chord module.
- Fix various unit tests bugs.
- Fix minor linter errors.



## 0.4.1 - 2021-06-04
Hotfix version according to issue [#23](https://github.com/Music-and-Culture-Technology-Lab/omnizart/issues/23)

## Feature
- Add a new piano transcription model and set it as the default model while using `music` module.

## Bugs
- Fix bug while parsing weight files in the checkpoint folder.

---

## 0.4.0 - 2021-05-31
Various improvements on music module and some critical bug fixes.

## Enhancement
- Improve the peak finding and thresholding strategy for more stable and better performance.
- Modify the feeding strategy of feature slices with adjustable overlapping rate while making predictions.
- Apply learning rate scheduler for music module.
- Replace the usage of custom training loop of music module with the built-in TF `.fit()` function.

## Bugs
- Fix a critical bug of inference of music module that would lead to missing onsets.
- Fix generation of pertubation of vocal module while training.

## Documentation
- Merge the demo page into master from `build_doc` branch.

---

## 0.3.4 - 2021-05-10
Hotifx version according to issue #19.

## Bugs
- Fix bug of treating numpy array as list while appending elements.

---

## 0.3.3 - 2021-05-07
Hotfix version according to issue #19.

## Bugs
- Fix column inconsistency of `aggregate_f0_info` and `write_agg_f0_results`.
- Update version of dependencies according to the security alert.

---

## 0.3.2 - 2021-02-13

### Enhancement
- Move `load_label` functions of different datasets into dataset structure classes.
- Add custom exception on fail downloading GD file due to access limit.
- Add unit tests on parsing label files into shared intermediate format.

### Bugs
- Fix wrong access name of the dict in vocal midi inference function.
- Fix bug of generating beat module training labels.

---

## 0.3.1 - 2021-01-18

Hotfix release of spleeter error.

### Bugs
- Call Spleeter in CLI mode instead of using python class.

---

## 0.3.0 - 2021-01-17

Release the `beat` module for symbolic domain beat transcription.

### Features
- Release `beat` module.
- Add an example `patch-cnn` module for demonstrating the implementation progress.

### Enhancement
- Refactor the flow of chord module for parsing the feature and label files.
- Modularize F0 information aggragation functions to *utils.py* and *io.py*.
- Improve verbosity on fail to open hdf files.

### Documentation
- Re-arrange the side bar with an additional group of CLI.
- Add custom CSS style for adjusting the width of audio and video elements.

### Bugs
- Fix Spleeter import errors after upgrading to v2.1.2.

---

## 0.2.0 - 2020-12-13

### Vocal melody transcription in both frame- and note-level are live!
We release the modules for vocal melody transcription after a decent amount of effort. 
Now you can transcribe your favorite singing voice.

### Features
- Release `vocal` and `vocal-contour` submodules.

### Enhancement
- Improve chord transcription results by filtering out chord predictions with short duration.
- Resolve the path for transcription output in a consistent way.

### Documentation
- Re-organize Quick Start and Tutorial pages to improve accessibility.
- Move the section for development from README.md to CONTRIBUTING.md.

### Bug Fix
- Fix bugs of passing the wrong parameter to vamp for chroma feature extraction.

---

## 0.1.1 - 2020-12-01
### Features
- Add more supported datasets for download and process.
- Supports to save checkpoints in .pb format with customized model checkpoint callback.

### Enhancement
- Huge refactor of constants.dataset. Improves reusability and add more useful common utilities.
- Modularize common parts of app classes.
- Construct base class of loading dataset samples. Reduce duplicate code and reuse the same functionalities.
- Filter out messy Tensorflow warnings when using CLI.

### Bug Fix
- Resolved bugs of some function parameters not actually being used inside functions.
- Fix CFP extraction down_fs don't actually work.

---

## 0.1.0 - 2020-11-16
### Features
- Add command for synthesizing MIDI file.
- Provides colab for quick start now!

### Enhancement
- Lazy import application instance for avoiding pulling large amount of dependencies.
- Group sub-commands into different sections when showing help message.

---

## 0.1.0-beta.2 - 2020-11-10

### Enhancement
- Better dealing with the input model path.
- Better approach for resolving dataset path when given with "./".
- Add documentation for Conda user for manually install omnizart.

### Bug Fix
- Fix wrong save path of checkpoints.
- Fix installation script for not upgrading pip after activating virtual environment.

---

## 0.1.0-beta.1 - 2020-11-08

First release of `omnizart` CLI tool, as well as a python package.

### Features
- Multi-instrument transcription
- Drum transcription
- Chord transcription
- Download datasets
- Extract feature of datasets for each module
- Train models for each module

---
