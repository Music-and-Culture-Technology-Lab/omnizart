# pylint: disable=R0201

import concurrent
from concurrent.futures import ProcessPoolExecutor

import scipy
import numpy as np
# Monkeypatch madmom DBNDownBeatTrackingProcessor to fix NumPy 2.0 inhomogeneous shape issue without modifying the virtual env
try:
    import madmom.features.downbeats
    import itertools as it

    def patched_process(self, activations, **kwargs):
        # use only the activations > threshold (init offset to be added later)
        first = 0
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]
        # return no beats if no activations given / remain after thresholding
        if not activations.any():
            return np.empty((0, 2))
        # (parallel) decoding of the activations with HMM
        results = list(self.map(madmom.features.downbeats._process_dbn, zip(self.hmms,
                                                   it.repeat(activations))))
        # choose the best HMM (highest log probability) - FIXED to avoid np.asarray inhomogeneous error
        best = np.argmax([r[1] for r in results])
        # the best path through the state space
        path, _ = results[best]
        # the state space and observation model of the best HMM
        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model
        # the positions inside the pattern (0..num_beats)
        positions = st.state_positions[path]
        # corresponding beats (add 1 for natural counting)
        beat_numbers = positions.astype(int) + 1
        if self.correct:
            beats = np.empty(0, dtype=int)
            # for each detection determine the "beat range", i.e. states where
            # the pointers of the observation model are >= 1
            beat_range = om.pointers[path] >= 1
            # get all change points between True and False (cast to int before)
            idx = np.nonzero(np.diff(beat_range.astype(int)))[0] + 1
            # if the first frame is in the beat range, add a change at frame 0
            if beat_range[0]:
                idx = np.r_[0, idx]
            # if the last frame is in the beat range, append the length of the
            # array
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            # iterate over all regions
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    # pick the frame with the highest activations value
                    # Note: we look for both beats and down-beat activations;
                    #       since np.argmax works on the flattened array, we
                    #       need to divide by 2
                    peak = np.argmax(activations[left:right]) // 2 + left
                    beats = np.hstack((beats, peak))
        else:
            # transitions are the points where the beat numbers change
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1
        # return the beat positions (converted to seconds) and beat numbers
        return np.vstack(((beats + first) / float(self.fps),
                          beat_numbers[beats])).T

    madmom.features.downbeats.DBNDownBeatTrackingProcessor.process = patched_process
except (ImportError, AttributeError):
    pass

from madmom.features import (
    DBNDownBeatTrackingProcessor,
    RNNDownBeatProcessor,
    DBNBeatTrackingProcessor,
    RNNBeatProcessor,
    BeatTrackingProcessor,
)

from omnizart.io import load_audio
from omnizart.utils import get_logger

logger = get_logger("Beat Extraction")


class MadmomBeatTracking:
    """Extract beat information with madmom library.

    Three different beat tracking methods are used together for producing a more
    stable beat tracking result.
    """
    def __init__(self, num_threads=3):
        self.num_threads = num_threads

    def _get_dbn_down_beat(self, audio_data_in1, min_bpm_in=50, max_bpm_in=230):
        proccesor = DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4, 5, 6, 7],
            min_bpm=min_bpm_in,
            max_bpm=max_bpm_in,
            fps=100,
            num_threads=self.num_threads
        )
        action = RNNDownBeatProcessor(num_threads=self.num_threads)(audio_data_in1)
        return proccesor(action)[:, 0]

    def _get_dbn_beat(self, audio_data_in2):
        proccesor = DBNBeatTrackingProcessor(fps=100, num_threads=self.num_threads)
        action = RNNBeatProcessor(num_threads=self.num_threads)(audio_data_in2)
        return proccesor(action)

    def _get_beat(self, audio_data_in3):
        proccesor = BeatTrackingProcessor(fps=100, num_threads=self.num_threads)
        action = RNNBeatProcessor(num_threads=self.num_threads)(audio_data_in3)
        return proccesor(action)

    def process(self, audio_data):
        """Generate beat tracking results with multiple approaches."""
        with ProcessPoolExecutor(max_workers=3) as executor:
            logger.debug("Submitting and executing parallel beat tracking jobs")
            future_1 = executor.submit(self._get_dbn_down_beat, audio_data, min_bpm_in=50, max_bpm_in=230)
            future_2 = executor.submit(self._get_dbn_beat, audio_data)
            future_3 = executor.submit(self._get_beat, audio_data)

            queue = {future_1: "dbn_down_beat", future_2: "dbn_beat", future_3: "beat"}

        results = {}
        for future in concurrent.futures.as_completed(queue, timeout=600):
            func_name = queue[future]
            results[func_name] = future.result()
            logger.debug("Job %s finished.", func_name)

        pred_beats1 = results["dbn_down_beat"]
        pred_beats2 = results["dbn_beat"]
        pred_beats3 = results["beat"]

        pred_beat_len1 = np.mean(
            np.sort(pred_beats1[1:] - pred_beats1[:-1])[int(len(pred_beats1) * 0.2):int(len(pred_beats1) * 0.8)]
        )
        pred_bpm1 = 60.0 / pred_beat_len1

        pred_beat_len2 = np.mean(
            np.sort(pred_beats2[1:] - pred_beats2[:-1])[int(len(pred_beats2) * 0.2):int(len(pred_beats2) * 0.8)]
        )
        pred_bpm2 = 60.0 / pred_beat_len2

        pred_beat_len3 = np.mean(
            np.sort(pred_beats3[1:] - pred_beats3[:-1])[int(len(pred_beats3) * 0.2):int(len(pred_beats3) * 0.8)]
        )
        pred_bpm3 = 60.0 / pred_beat_len3
        pred_bpm_avg = np.mean([pred_bpm1, pred_bpm2, pred_bpm3])

        logger.debug("Running last beat tracking step...")
        return self._get_dbn_down_beat(audio_data, min_bpm_in=pred_bpm_avg / 1.38, max_bpm_in=pred_bpm_avg * 1.38)


def extract_beat_with_madmom(audio_path, sampling_rate=44100):
    """Extract beat position (in seconds) of the audio.

    Extract beat with mixture of beat tracking techiniques using madmom.

    Parameters
    ----------
    audio_path: Path
        Path to the target audio
    sampling_rate: int
        Desired sampling to be resampled.

    Returns
    -------
    beat_arr: 1D numpy array
        Contains beat positions in seconds.
    audio_len_sec: float
        Total length of the audio in seconds.
    """
    logger.debug("Loading audio: %s", audio_path)
    audio_data, _ = load_audio(audio_path, sampling_rate=sampling_rate)
    logger.debug("Runnig beat tracking...")
    return MadmomBeatTracking().process(audio_data), len(audio_data) / sampling_rate


def extract_mini_beat_from_beat_arr(beat_arr, audio_len_sec, mini_beat_div_n=32):
    """Extract mini beats from the beat array.

    Furhter split beat into shorter beat interval, which we call it *mini beat*, to increase the
    beat resolution. We use linear interpolation to generate the mini beats.

    Parameters
    ----------
    beat_arr: 1D numpy array
        Beat array generated by `extract_beat_with_madmom`.
    audio_len_sec: float
        Total length of the audio in seconds.
    mini_beat_div_n: int
        Number of mini beats in a single 4/4 measure.

    Returns
    -------
    mini_beat_pos_t: 1D numpy array
        Positions of mini beats in seconds.

    """
    # How many division per bar on the 4/4 time signiture basis
    mini_beat_div_n = np.round(mini_beat_div_n / 4).astype("int")

    beat_time_ary_in = np.array(beat_arr)
    beat_abs_idx = np.arange(len(beat_arr)) + 1  # count from 1
    beat_map_func = scipy.interpolate.interp1d(beat_abs_idx, beat_time_ary_in, fill_value="extrapolate")

    mini_beat_abs_idx = np.arange(0, beat_arr.shape[0] + 1, (1 / mini_beat_div_n))
    mini_beat_pos_t = [beat_map_func(x) for x in mini_beat_abs_idx]

    # Filter out beat outside audio time range
    mini_beat_pos_t = np.array([x for x in mini_beat_pos_t if x >= 0])
    mini_beat_pos_t = np.array([x for x in mini_beat_pos_t if x <= audio_len_sec])

    return mini_beat_pos_t


def extract_mini_beat_from_audio_path(audio_path, sampling_rate=44100, mini_beat_div_n=32):
    """ Wrapper of extracting mini beats from audio path. """
    logger.debug("Extracting beat with madmom")
    beat_arr, audio_len_sec = extract_beat_with_madmom(audio_path, sampling_rate=sampling_rate)
    logger.debug("Extracting mini beat")
    return extract_mini_beat_from_beat_arr(beat_arr, audio_len_sec, mini_beat_div_n=mini_beat_div_n)


if __name__ == "__main__":
    AUDIO_PATH = "checkpoints/Last Stardust - piano.wav"
    mini_beat_arr = extract_mini_beat_from_audio_path(AUDIO_PATH)
