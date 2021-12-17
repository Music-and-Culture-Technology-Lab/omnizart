# OMNIZART

[![build](https://github.com/Music-and-Culture-Technology-Lab/omnizart/workflows/general-check/badge.svg)](https://github.com/Music-and-Culture-Technology-Lab/omnizart/actions?query=workflow%3Ageneral-check)
[![docs](https://github.com/Music-and-Culture-Technology-Lab/omnizart/workflows/docs/badge.svg?branch=build_doc)](https://music-and-culture-technology-lab.github.io/omnizart-doc/)
[![PyPI version](https://badge.fury.io/py/omnizart.svg)](https://badge.fury.io/py/omnizart)
![PyPI - License](https://img.shields.io/pypi/l/omnizart)
[![Downloads](https://static.pepy.tech/personalized-badge/omnizart?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads/month)](https://pepy.tech/project/omnizart)
[![Docker Pulls](https://img.shields.io/docker/pulls/mctlab/omnizart)](https://hub.docker.com/r/mctlab/omnizart)

[![DOI](https://joss.theoj.org/papers/10.21105/joss.03391/status.svg)](https://doi.org/10.21105/joss.03391)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5769022.svg)](https://doi.org/10.5281/zenodo.5769022)


Omnizart is a Python library that aims for democratizing automatic music transcription.
Given polyphonic music, it is able to transcribe pitched instruments, vocal melody, chords, drum events, and beat.
This is powered by the research outcomes from [Music and Culture Technology (MCT) Lab](https://sites.google.com/view/mctl/home). The paper has been published to [Journal of Open Source Software (JOSS)](https://doi.org/10.21105/joss.03391).

### Transcribe your favorite songs now in Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/OmnizartColab) or [![Replicate](https://replicate.com/breezewhite/omnizart/badge)](https://replicate.ai/breezewhite/omnizart)

# Quick start

Visit the [complete document](https://music-and-culture-technology-lab.github.io/omnizart-doc/) for detailed guidance.

## Pip
``` bash
# Install omnizart
pip install omnizart

# Download the checkpoints
omnizart download-checkpoints

# Transcribe your songs
omnizart drum transcribe <path/to/audio.wav>
omnizart chord transcribe <path/to/audio.wav>
omnizart music transcribe <path/to/audio.wav>
```

## Docker
```
docker pull mctlab/omnizart:latest
docker run -it mctlab/omnizart:latest bash
```

# Supported applications
| Application      | Transcription      | Training           | Evaluation | Description                                      |
|------------------|--------------------|--------------------|------------|--------------------------------------------------|
| music            | :heavy_check_mark: | :heavy_check_mark: |            | Transcribe musical notes of pitched instruments. |
| drum             | :heavy_check_mark: | :interrobang:      |            | Transcribe events of percussive instruments.     |
| vocal            | :heavy_check_mark: | :heavy_check_mark: |            | Transcribe note-level vocal melody.              |
| vocal-contour    | :heavy_check_mark: | :heavy_check_mark: |            | Transcribe frame-level vocal melody (F0).        |
| chord            | :heavy_check_mark: | :heavy_check_mark: |            | Transcribe chord progressions.                   |
| beat             | :heavy_check_mark: | :heavy_check_mark: |            | Transcribe beat position.                        |

**NOTES**
The current implementation for the drum model has unknown bugs, preventing loss convergence when training from scratch.
Fortunately, you can still enjoy drum transcription with the provided checkpoints.

## Compatibility Issue
Currently, Omnizart is **incompatible for ARM-based MacOS** system due to the underlying dependencies.
More details can be found in the [issue #38](https://github.com/Music-and-Culture-Technology-Lab/omnizart/issues/38).

## Citation
If you use this software in your work, please cite:

```
@article{Wu2021,
  doi = {10.21105/joss.03391},
  url = {https://doi.org/10.21105/joss.03391},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {68},
  pages = {3391},
  author = {Yu-Te Wu and Yin-Jyun Luo and Tsung-Ping Chen and I-Chieh Wei and Jui-Yang Hsu and Yi-Chin Chuang and Li Su},
  title = {Omnizart: A General Toolbox for Automatic Music Transcription},
  journal = {Journal of Open Source Software}
}
```
