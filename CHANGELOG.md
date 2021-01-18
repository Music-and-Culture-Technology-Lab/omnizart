# Changelog

## 0.3.1 - 2020-01-18

### Bugs
- Call Spleeter in CLI mode instead of using python class.

## 0.3.0 - 2020-01-17

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
