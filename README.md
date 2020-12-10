# omnizart

[![build](https://github.com/Music-and-Culture-Technology-Lab/omnizart/workflows/general-check/badge.svg)](https://github.com/Music-and-Culture-Technology-Lab/omnizart/actions?query=workflow%3Ageneral-check)
[![docs](https://github.com/Music-and-Culture-Technology-Lab/omnizart/workflows/docs/badge.svg?branch=build_doc)](https://music-and-culture-technology-lab.github.io/omnizart-doc/)
[![PyPI version](https://badge.fury.io/py/omnizart.svg)](https://badge.fury.io/py/omnizart)
![PyPI - License](https://img.shields.io/pypi/l/omnizart)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/omnizart)](https://pypistats.org/packages/omnizart)
[![Docker Pulls](https://img.shields.io/docker/pulls/mctlab/omnizart)](https://hub.docker.com/r/mctlab/omnizart)

Omniscient Mozart, being able to transcribe everything in the music, including vocal, drum, chord, beat, instruments, and more.
Combines all the hard works developed by everyone in MCTLab into a single command line tool. Python package and docker
image are also available.

### Try omnizart now!! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/omnizart-colab)

A quick-start example is as following:
``` bash
# Install omnizart
pip install omnizart

# Download the checkpoints after installation
omnizart download-checkpoints

# Now it's ready for the transcription~
omnizart drum transcribe <path/to/audio.wav>
omnizart chord transcribe <path/to/audio.wav>
omnizart music transcribe <path/to/audio.wav>
```

Or use the docker image:
```
docker pull mctlab/omnizart:latest
docker run -it mctlab/omnizart:latest bash
```

Comprehensive usage and API references can be found in the [official documentation site](https://music-and-culture-technology-lab.github.io/omnizart-doc/).

# About
[Music and Culture Technology Lab (MCTLab)](https://sites.google.com/view/mctl/home) aims to develop technology for music and relevant applications by leveraging cutting-edge AI techiniques.

# Plan to support
| Commands         | transcribe         | train              | evaluate | Description                       |
|------------------|--------------------|--------------------|----------|-----------------------------------|
| music            | :heavy_check_mark: | :heavy_check_mark: |          | Transcribes notes of instruments. |
| drum             | :heavy_check_mark: | :interrobang:      |          | Transcribes drum tracks.          |
| vocal            |                    |                    |          | Transcribes pitch of vocal.       |
| vocal-contour    |                    |                    |          | Transcribes contour of vocal.     |
| chord            | :heavy_check_mark: | :heavy_check_mark: |          | Transcribes chord progression.    |
| beat             |                    |                    |          | Transcribes beat position.        |

**NOTES** Though the implementation of training the drum model is 90% complete, but there still exists some
invisible bugs that cause the training fails to converge compared to the author's original implementation.

Example usage
<pre>
omnizart music transcribe <i>path/to/audio</i>
omnizart chord transcribe <i>path/to/audio</i>
omnizart drum transcribe <i>path/to/audio</i>
</pre>

For training a new model, download the dataset first and follow steps described below.
<pre>
# The following command will default saving the extracted feature under the same folder,
# called <b>train_feature</b> and <b>test_feature</b>
omnizart music generate-feature -d <i>path/to/dataset</i>

# Train a new model
omnizart music train-model -d <i>path/to/dataset</i>/train_feature --model-name My-Model
</pre>
