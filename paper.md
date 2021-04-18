---
title: 'Omnizart: A General Toolbox for Automatic Music Transcription'
tags:
  - Python 
  - automatic music transcription
  - music information retrieval
  - audio signal processing
  - artificial intelligence
authors:
  - name: Yu-Te Wu 
    affiliation: 1
  - name: Yin-Jyun Luo
    affiliation: 1
  - name: Tsung-Ping Chen
    affiliation: 1
  - name: I-Chieh Wei
    affiliation: 1
  - name: Jui-Yang Hsu
    affiliation: 1
  - name: Yi-Chin Chuang
    affiliation: 1
  - name: Li Su
    affiliation: 1
affiliations:
 - name: Music and Culture Technology Lab, Institute of Information Science, Academia Sinica, Taipei, Taiwan
   index: 1
date: 18 April 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---


# Summary

We present and release Omnizart, a new Python library that provides a streamlined solution to automatic music transcription (AMT).
Omnizart encompasses modules that construct the life-cycle of deep learning-based AMT, and is designed for ease of use with a compact command-line interface.
To the best of our knowledge, Omnizart is the first transcription toolkit which offers models covering a wide class of instruments ranging from solo, instrument ensembles, percussion instruments to vocal, as well as models for chord recognition and beat/downbeat tracking, two music information retrieval (MIR) tasks highly related to AMT. 
In summary, Omnizart incorporates:

- Pre-trained models for frame-level and note-level transcription of multiple pitched instruments, vocal melody, and drum events;
- Pre-trained models of chord recognition and beat/downbeat tracking;
- Main functionalities in the life-cycle of AMT research, covering from dataset downloading, feature pre-processing, model training, to sonification of the transcription result.

Omnizart is based on Tensorflow @[abadi2016tensorflow]. 
The complete code base, command-line interface, documentation, as well as demo examples can all be accessed from the [project website](https://github.com/Music-and-Culture-Technology-Lab/omnizart).


# Purpose

AMT has been one of the core challenges in MIR because of the multifaceted nature of musical signals. 
Typically, streams of musical notes performed with various instruments overlap with each other and then create a hierarchy of abstraction. This complicates the task to identify the melodic, timbral, and rhythmic attributes of the music. 

While the majority of the previous solution focuses on single-instrument transcription, Omnizart collects several state-of-tha-art (SoTA) models for transcribing multiple pitched and percussive instruments, as well as vocal out of the interference with rich music polyphony.
Omnizart also finds it applicability for chord recognition and beat tracking.
As such, the proposed library offers a unified solution to music transcription for multi-track and modalities.
Additionally, researchers can leverage the transcribed outputs as supervisory resources to approach other MIR tasks such as source separation and music generation.

Omnizart provides pre-trained models that closely reproduce the performances reported in the original papers, which can be fine-tuned with separate datasets for benchmarking purposes.
We believe the release of Omnizart can accelerate the advance of AMT research and contribute to the MIR community.


# Implementation Details

## Piano solo transcription

The piano solo transcription model in Omnizart reproduces the implementation of @[wu2020multi].
The model features a U-net which takes as inputs the audio spectrogram, generalized cepstrum (GC) @[su2015combining], and GC of spectrogram (GCoS) @[wu2018automatic], and outputs a multi-channel time-pitch representation with time- and pitch-resolution of 20ms and 25 cents, respectively.
For the U-net, implementation of the encoder and the decoder follows DeepLabV3+ @[Chen2018DeepLabV3], and the bottleneck layer is adapted from the Image Transformer @[parmar2018image].

The model is trained on the MAESTRO dataset @[hawthorne2018enabling], an external dataset containing 1,184 real piano performance recordings with a total length of 172.3 hours.
The model achieves 72.50\% and 79.57\% for frame- and note-level F1-scores, respectively, on the Configuration-II test set of the MAPS dataset @[kelz2016potential].

## Multi-instrument polyphonic transcription

The multi-instrument transcription model extends the piano solo model to support 11 output classes, namely piano, violin, viola, cello, flute, horn, bassoon, clarinet, harpsichord, contrabass, and oboe, accessed from MusicNet @[thickstun2017learning].
Notably, the model allows for \emph{instrument-agnostic transcription} where the instruments to transcribe are unknown during inference @[wu2020multi].
The evaluation on the test set from MusicNet @[thickstun2018invariances] yields 66.59\% for the note streaming task.

## Drum transcription

The model for drum transcription is a re-implementation of @[wei2021improving] which predicts percussive events from a given input audio.
Building blocks of the network include convolutional layers and the attention mechanism.

The model is trained on a dataset with 1,454 audio clips of polyphonic music with synchronized drum events @[wei2021improving].
The model demonstrates SoTA performance on two commonly used benchmark datasets, i.e., 74\% for ENST @[gillet2006enst] and 71\% for MDB-Drums @[southall2017mdb] in terms of the note-level F1-score.

## Vocal transcription in polyphonic music

The system for vocal transcription features a pitch extractor and a module for note segmentation.
The inputs to the model are composed of spectrogram, GS, and GCoS derived from polyphonic music recordings @[wu2018automatic].

A pre-trained Patch-CNN @[su2018vocal] is leveraged as the pitch extractor.
The module for note segmentation is implemented with PyramidNet-110 and ShakeDrop regularization @[yamada2019shakedrop], which is trained using Virtual Adversarial Training @[miyato2018virtual] enabling semi-supervised learning.

The training includes labeled data from TONAS @[mora2010characterization] and unlabeled ones from MIR-1K @[hsu2009improvement].
The model yields the SoTA F1-score of 68.4\% evaluated with the ISMIR2014 dataset @[molina2014evaluation].


## Chord recognition

The harmony recognition function of Omnizart is implemented using the Harmony Transformer (HT) @[chen2019harmony]. 
The HT model is based on an encoder-decoder architecture,
where the encoder performs chord segmentation on the input, and the decoder recognizes the chord progression based on the segmentation result.

The original HT supports both audio and symbolic inputs.
Currently, Omnizart supports only audio inputs.
A given audio input is pre-processed using Chordino VAMP plugin @[mauch2010approximate] as the non-negative-least-squares chromagram.
The outputs of the model include 25 chord types, covering 12 major and minor chords together with a class referred to the absence of chord, with a time resolution of 230ms.

In an experiment with evaluations on the McGill Billboard dataset @[burgoyne2011anexpert], the HT outperforms the previous SoTAs @[chen2019harmony]. 

## Beat/downbeat tracking

The model for beat and downbeat tracking provided in Omnizart is a reproduction of @[chuang2020beat].
Unlike most of the available open-source projects such as \texttt{madmom} @[bock2016madmom] and \texttt{librosa} @[mcfee2015librosa] which focus on audio, the provided model targets symbolic data.

The input and output of the model are respectively MIDI and beat/downbeat positions with the time resolution of 10ms.
The input representation combines piano-roll, spectral flux, and inter-onset interval extracted from MIDI.
The model composes a two-layer BLSTM network with the attention mechanism, and predict probabilities of the presence of beat and downbeat per time step.

Experiments on the MusicNet dataset @[thickstun2018invariances] with the synchronized beat annotation show that the proposed model outperforms the SoTA beat trackers which operate on synthesized audio @[chuang2020beat].  

# Conclusion

Omnizart represents the first systematic solution for the polyphonic AMT of general music contents ranging from pitched instruments, percussion instrument, to voice. 
In addition to note transcription, Omnizart also includes high-level MIR tasks such as chord recognition and beat/downbeat tracking.
As an ongoing project, the research group will keep refining the package and also extending the scope of transcription in the future.

# References
