---
title: "Omnizart: A General Toolbox for Automatic Music Transcription"
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
To the best of our knowledge, Omnizart is the first toolkit that offers transcription models for various music content including piano solo, instrument ensembles, percussion and vocal. Omnizart also supports models for chord recognition and beat/downbeat tracking, which are highly related to AMT.

In summary, Omnizart incorporates:

- Pre-trained models for frame-level and note-level transcription of multiple pitched instruments, vocal melody, and drum events;
- Pre-trained models of chord recognition and beat/downbeat tracking;
- The main functionalities in the life-cycle of AMT research, covering dataset downloading, feature pre-processing, model training, to the sonification of the transcription result.

Omnizart is based on Tensorflow [@abadi2016tensorflow].
The complete code base, command-line interface, documentation, as well as demo examples can all be accessed from the [project website](https://github.com/Music-and-Culture-Technology-Lab/omnizart).

# Statement of need

AMT of polyphonic music is a complicated MIR task because the note-, melody-, timbre-, and rhythm-level attributes of music are overlapped with each other in music signals. A unified solution of AMT is therefore in eager demand. AMT is also strongly related to other MIR tasks such as source separation and music generation with transcribed data needed as supervisory resources.
Omnizart considers multi-instrument transcription and collects several state-of-the-art models for transcribing pitched and percussive instruments, as well as singing voice, within polyphonic music signals. Omnizart is an AMT tool that unifies multiple transcription utilities and enables further productivity. Omnizart can save one's time and labor in generating a massive number of multi-track MIDI files, which could have a large impact on music production, music generation, education, and musicology research.

# Implementation Details

## Piano solo transcription

The piano solo transcription model in Omnizart reproduces the implementation of @wu2020multi.
The model features a U-net that takes as inputs the audio spectrogram, generalized cepstrum (GC) [@su2015combining], and GC of spectrogram (GCoS) [@wu2018automatic], and outputs a multi-channel time-pitch representation with time- and pitch-resolution of 20 ms and 25 cents, respectively.
For the U-net, implementation of the encoder and the decoder follows DeepLabV3+ [@Chen2018DeepLabV3], and the bottleneck layer is adapted from the Image Transformer [@parmar2018image].

The model is trained on the MAESTRO dataset [@hawthorne2018enabling], an external dataset containing 1,184 real piano performance recordings with a total length of 172.3 hours.
The model achieves 72.50\% and 79.57\% for frame- and note-level F1-scores, respectively, on the Configuration-II test set of the MAPS dataset [@kelz2016potential].

## Multi-instrument polyphonic transcription

The multi-instrument transcription model extends the piano solo model to support 11 output classes, namely piano, violin, viola, cello, flute, horn, bassoon, clarinet, harpsichord, contrabass, and oboe, accessed from MusicNet [@thickstun2017learning].
Detailed characteristics of the model can be seen in @wu2020multi.
The evaluation on the test set from MusicNet [@thickstun2018invariances] yields 66.59\% for the note streaming task.

## Drum transcription

The model for drum transcription is a re-implementation of @wei2021improving.
Building blocks of the network include convolutional layers and the attention mechanism.

The model is trained on a dataset with 1,454 audio clips of polyphonic music with synchronized drum events [@wei2021improving].
The model demonstrates SoTA performance on two commonly used benchmark datasets, i.e., 74\% for ENST [@gillet2006enst] and 71\% for MDB-Drums [@southall2017mdb] in terms of the note-level F1-score.

## Vocal transcription in polyphonic music

The system for vocal transcription features a pitch extractor and a module for note segmentation.
The inputs to the model are composed of spectrogram, GS, and GCoS derived from polyphonic music recordings [@wu2018automatic].

A pre-trained Patch-CNN [@su2018vocal] is leveraged as the pitch extractor.
The module for note segmentation is implemented with PyramidNet-110 and ShakeDrop regularization [@yamada2019shakedrop], which is trained using Virtual Adversarial Training [@miyato2018virtual] enabling semi-supervised learning.

The training data includes labeled data from TONAS [@mora2010characterization] and unlabeled data from MIR-1K [@hsu2009improvement].
The model yields the SoTA F1-score of 68.4\% evaluated with the ISMIR2014 dataset [@molina2014evaluation].

## Chord recognition

The harmony recognition model of Omnizart is implemented using the Harmony Transformer (HT) [@chen2019harmony].
The HT model is based on an encoder-decoder architecture,
where the encoder performs chord segmentation on the input, and the decoder recognizes the chord progression based on the segmentation result.

The original HT supports both audio and symbolic inputs.
Currently, Omnizart supports only audio inputs.
A given audio input is pre-processed using Chordino VAMP plugin [@mauch2010approximate] as the non-negative-least-squares chromagram.
The outputs of the model include 25 chord types, covering 12 major and minor chords together with a class referred to the absence of chord, with a time resolution of 230 ms.

In an experiment with evaluations on the McGill Billboard dataset [@burgoyne2011anexpert], the HT outperforms the previous state of the art [@chen2019harmony].

## Beat/downbeat tracking

The model for beat and downbeat tracking provided in Omnizart is a reproduction of @chuang2020beat.
Unlike most of the available open-source projects such as \texttt{madmom} [@bock2016madmom] and \texttt{librosa} [@mcfee2015librosa] which focus on audio, the provided model targets symbolic data.

The input and output of the model are respectively MIDI and beat/downbeat positions with the time resolution of 10 ms.
The input representation combines piano-roll, spectral flux, and inter-onset interval extracted from MIDI.
The model composes a two-layer BLSTM network with the attention mechanism, and predicts probabilities of the presence of beat and downbeat per time step.

Experiments on the MusicNet dataset [@thickstun2018invariances] with the synchronized beat annotation show that the proposed model outperforms the state-of-the-art beat trackers which operate on synthesized audio [@chuang2020beat].

# Conclusion

Omnizart represents the first systematic solution for the polyphonic AMT of general music contents ranging from pitched instruments, percussion instruments, to voices.
In addition to note transcription, Omnizart also includes high-level MIR tasks such as chord recognition and beat/downbeat tracking.
As an ongoing project, the research group will keep refining the package and extending the scope of transcription in the future.

# References
