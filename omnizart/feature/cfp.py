# -*- coding: utf-8 -*-
"""
Author: Lisu

Mantainer: BreezeWhite
"""
# pylint: disable=C0103,W0102,R0914
import numpy as np
import scipy

from omnizart.utils import load_audio_with_librosa, get_logger, parallel_generator


logger = get_logger("CFP Feature")


def STFT(x, fr, fs, Hop, h):
    t = np.arange(Hop, np.ceil(len(x) / float(Hop)) * Hop, Hop)
    N = int(fs / float(fr))
    window_size = len(h)
    f = fs * np.linspace(0, 0.5, np.round(N / 2).astype("int"), endpoint=True)
    Lh = int(np.floor(float(window_size - 1) / 2))
    tfr = np.zeros((int(N), len(t)), dtype=np.float)

    for icol, ti in enumerate(t):
        ti = int(ti)
        tau = np.arange(int(-min([round(N / 2.0) - 1, Lh, ti - 1])), int(min([round(N / 2.0) - 1, Lh, len(x) - ti])))
        indices = np.mod(N + tau, N) + 1
        tfr[indices - 1, icol] = x[ti + tau - 1] * h[Lh + tau - 1] / np.linalg.norm(h[Lh + tau - 1])

    tfr = abs(scipy.fftpack.fft(tfr, n=N, axis=0))
    return tfr, f, t, N


def nonlinear_func(X, g, cutoff):
    cutoff = int(cutoff)
    if g != 0:
        X[X < 0] = 0
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
        X = np.power(X, g)
    else:
        X = np.log(X)
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
    return X


def freq_to_log_freq_mapping(tfr, f, fr, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1 / tc
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        cen_freq = StartFreq * pow(2, float(i) / NumPerOct)
        if cen_freq < StopFreq:
            central_freq.append(cen_freq)
        else:
            break

    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)
    for i in range(1, Nest - 1):
        left = int(round(central_freq[i - 1] / fr))
        right = int(round(central_freq[i + 1] / fr) + 1)

        # rounding1
        if left >= right - 1:
            freq_band_transformation[i, left] = 1
        else:
            for j in range(left, right):
                if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                    freq_band_transformation[i, j] = (f[j] - central_freq[i-1]) / (central_freq[i] - central_freq[i-1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                    freq_band_transformation[i, j] = (central_freq[i+1] - f[j]) / (central_freq[i+1] - central_freq[i])
    tfrL = np.dot(freq_band_transformation, tfr)
    return tfrL, central_freq


def quef_to_log_freq_mapping(ceps, q, fs, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1 / tc
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        cen_freq = StartFreq * pow(2, float(i) / NumPerOct)
        if cen_freq < StopFreq:
            central_freq.append(cen_freq)
        else:
            break
    f = 1 / (q+1e-9)
    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)
    for i in range(1, Nest - 1):
        for j in range(int(round(fs / central_freq[i + 1])), int(round(fs / central_freq[i - 1]) + 1)):
            if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                freq_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (central_freq[i] - central_freq[i - 1])
            elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])

    tfrL = np.dot(freq_band_transformation[:, :len(ceps)], ceps)
    return tfrL, central_freq


def cfp_filterbank(x, fr, fs, Hop, h, fc, tc, g, bin_per_octave):
    NumofLayer = np.size(g)

    [tfr, f, t, N] = STFT(x, fr, fs, Hop, h)
    tfr = np.power(abs(tfr), g[0])
    tfr0 = tfr  # original STFT
    ceps = np.zeros(tfr.shape)

    if NumofLayer >= 2:
        for gc in range(1, NumofLayer):
            if np.remainder(gc, 2) == 1:
                tc_idx = round(fs * tc)
                ceps = np.real(np.fft.fft(tfr, axis=0)) / np.sqrt(N)
                ceps = nonlinear_func(ceps, g[gc], tc_idx)
            else:
                fc_idx = round(fc / fr)
                tfr = np.real(np.fft.fft(ceps, axis=0)) / np.sqrt(N)
                tfr = nonlinear_func(tfr, g[gc], fc_idx)

    tfr0 = tfr0[:int(round(N / 2)), :]
    tfr = tfr[:int(round(N / 2)), :]
    ceps = ceps[:int(round(N / 2)), :]

    HighFreqIdx = int(round((1/tc) / fr) + 1)
    f = f[:HighFreqIdx]
    tfr0 = tfr0[:HighFreqIdx, :]
    tfr = tfr[:HighFreqIdx, :]
    HighQuefIdx = int(round(fs / fc) + 1)
    q = np.arange(HighQuefIdx) / float(fs)
    ceps = ceps[:HighQuefIdx, :]

    tfrL0, central_frequencies = freq_to_log_freq_mapping(tfr0, f, fr, fc, tc, bin_per_octave)
    tfrLF, central_frequencies = freq_to_log_freq_mapping(tfr, f, fr, fc, tc, bin_per_octave)
    tfrLQ, central_frequencies = quef_to_log_freq_mapping(ceps, q, fs, fc, tc, bin_per_octave)

    return tfrL0, tfrLF, tfrLQ, f, q, t, central_frequencies


def parallel_extract(x, samples, max_sample, fr, fs, Hop, h, fc, tc, g, bin_per_octave):
    freq_width = max_sample * Hop
    iters = np.ceil(samples / max_sample).astype("int")
    tmpL0, tmpLF, tmpLQ, tmpZ = {}, {}, {}, {}

    slice_list = [x[i * freq_width:(i+1) * freq_width] for i in range(iters)]

    feat_generator = enumerate(
        parallel_generator(
            cfp_filterbank,
            slice_list,
            fr=fr,
            fs=fs,
            Hop=Hop,
            h=h,
            fc=fc,
            tc=tc,
            g=g,
            bin_per_octave=bin_per_octave,
            max_workers=3)
    )
    for idx, (feat_list, slice_idx) in feat_generator:
        logger.debug("Slice feature extracted: %d/%d", idx+1, len(slice_list))
        tfrL0, tfrLF, tfrLQ, f, q, t, cen_freq = feat_list
        tmpL0[slice_idx] = tfrL0
        tmpLF[slice_idx] = tfrLF
        tmpLQ[slice_idx] = tfrLQ
        tmpZ[slice_idx] = tfrLF * tfrLQ
    return tmpL0, tmpLF, tmpLQ, tmpZ, f, q, t, cen_freq


def extract_cfp(
    filename,
    hop=0.02,  # in seconds
    win_size=7939,
    fr=2.0,
    fc=27.5,
    tc=1/4487.0,
    g=[0.24, 0.6, 1],
    bin_per_octave=48,
    down_fs=44100,
    max_sample=2000,
):
    """CFP feature extraction function.

    Given the audio path, returns the CFP feature.

    Returns
    -------
    Z
        Multiplication of spectrum and cepstrum
    tfrL0
        Spectrum of the audio.
    tfrLF
        Generalized Cepstrum of Spectrum (GCoS).
    tfrLQ
        Cepstrum of the audio
    cen_freq
        Central frequencies of the feature dimension.

    References
    ----------
    The CFP approach was first proposed in [1]_

    .. [1] L. Su and Y. Yang, "Combining Spectral and Temporal Representations for Multipitch Estimation of Polyphonic
       Music," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2015.
    """

    logger.debug("Loading audio: %s", filename)
    x, fs = load_audio_with_librosa(filename, sampling_rate=down_fs)
    Hop = round(down_fs * hop)
    x = x.astype("float32")
    h = scipy.signal.blackmanharris(win_size)  # window size
    g = np.array(g)

    max_sample = 2000
    samples = np.floor(len(x) / Hop).astype("int")
    logger.debug("Sample number: %d", samples)
    logger.debug("Extracting CFP feature...")
    if samples > max_sample:
        tmpL0, tmpLF, tmpLQ, tmpZ, _, _, _, cen_freq = parallel_extract(
            x, samples, max_sample, fr, fs, Hop, h, fc, tc, g, bin_per_octave
        )

        tfrL0 = tmpL0.pop(0)
        tfrLF = tmpLF.pop(0)
        tfrLQ = tmpLQ.pop(0)
        Z = tmpZ.pop(0)
        rr = len(tmpL0)
        for i in range(1, rr + 1, 1):
            tfrL0 = np.concatenate((tfrL0, tmpL0.pop(i)), axis=1)
            tfrLF = np.concatenate((tfrLF, tmpLF.pop(i)), axis=1)
            tfrLQ = np.concatenate((tfrLQ, tmpLQ.pop(i)), axis=1)
            Z = np.concatenate((Z, tmpZ.pop(i)), axis=1)
    else:
        tfrL0, tfrLF, tfrLQ, _, _, _, cen_freq = cfp_filterbank(x, fr, fs, Hop, h, fc, tc, g, bin_per_octave)
        Z = tfrLF * tfrLQ

    return Z, tfrL0, tfrLF, tfrLQ, cen_freq
