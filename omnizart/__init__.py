import os
import collections
import collections.abc

# Patch collections module for backward compatibility with old libraries under Python 3.10+
for name in ['MutableSequence', 'Iterable', 'Mapping', 'Sequence', 'Callable', 'Container', 'MutableMapping']:
    if not hasattr(collections, name) and hasattr(collections.abc, name):
        setattr(collections, name, getattr(collections.abc, name))

# Patch numpy for backward compatibility with old libraries under NumPy 2.0+
import numpy as np
for name, target in [('float', float), ('int', int), ('bool', bool), ('complex', complex)]:
    if not hasattr(np, name):
        setattr(np, name, target)

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

MODULE_PATH = os.path.abspath(f"{__file__}/..")
SETTING_DIR = os.path.join(MODULE_PATH, "defaults")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['VAMP_PATH'] = os.path.join(MODULE_PATH, "resource", "vamp")

__version__ = "0.5.0"
