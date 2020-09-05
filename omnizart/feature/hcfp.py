# pylint: disable=C0103,W0102,R0914

import numpy as np

from omnizart.feature.cfp import extract_cfp
from omnizart.utils import get_logger


logger = get_logger("HCFP Feature")


def fetch_harmonic(data, cenf, ith_har, start_freq=27.5, num_per_octave=48, is_reverse=False):
    ith_har += 1
    if ith_har != 0 and is_reverse:
        ith_har = 1 / ith_har

    # harmonic_series = [12, 19, 24, 28, 31]
    bins_per_note = int(num_per_octave / 12)
    total_bins = int(bins_per_note * 88)

    hid = min(range(len(cenf)), key=lambda i: abs(cenf[i] - ith_har*start_freq))  # noqa: E226

    harmonic = np.zeros((total_bins, data.shape[1]))
    upper_bound = min(len(cenf) - 1, hid + total_bins)
    harmonic[:(upper_bound - hid)] = data[hid:upper_bound]

    return harmonic


def extract_hcfp(
    filename,
    hop=0.02,  # in seconds
    win_size=7939,
    fr=2.0,
    g=[0.24, 0.6, 1],
    bin_per_octave=48,
    down_fs=44100,
    max_sample=2000,
    harmonic_num=6,
):
    _, spec, gcos, ceps, cenf = extract_cfp(
        filename,
        hop=hop,
        win_size=win_size,
        fr=fr,
        fc=1.0,
        tc=1 / 22050,
        g=g,
        bin_per_octave=bin_per_octave,
        down_fs=down_fs,
        max_sample=max_sample,
    )

    har = []
    logger.debug("Fetching harmonics of spectrum")
    for i in range(harmonic_num + 1):
        har.append(fetch_harmonic(spec, cenf, i))
    har_s = np.transpose(np.array(har), axes=(2, 1, 0))

    # Harmonic GCoS
    har = []
    logger.debug("Fetching harmonics of GCoS")
    for i in range(harmonic_num + 1):
        har.append(fetch_harmonic(gcos, cenf, i))
    har_g = np.transpose(np.array(har), axes=(2, 1, 0))

    # Harmonic cepstrum
    har = []
    logger.debug("Fetching harmonics of cepstrum")
    for i in range(harmonic_num + 1):
        har.append(fetch_harmonic(ceps, cenf, i, is_reverse=True))
    har_c = np.transpose(np.array(har), axes=(2, 1, 0))

    return har_s, har_g, har_c, cenf
