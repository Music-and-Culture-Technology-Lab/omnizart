import numpy as np
from scipy.special import expit
from librosa.core import midi_to_hz

from omnizart.constants.midi import LOWEST_MIDI_NOTE


def inference(feature, model, timestep=128, batch_size=10, feature_num=384):
    assert len(feature.shape) == 2
    # Padding
    total_samples = len(feature)
    pad_bottom = (feature_num - feature.shape[1]) // 2
    pad_top = feature_num - feature.shape[1] - pad_bottom
    pad_len = timestep - 1
    feature = np.pad(feature, ((pad_len, pad_len), (pad_bottom, pad_top)))

    # Prepare for prediction
    output = np.zeros(feature.shape + (2,))
    total_batches = int(np.ceil(total_samples / batch_size))
    last_batch_idx = len(feature) - pad_len
    for bidx in range(total_batches):
        print(f"batch: {bidx+1}/{total_batches}", end="\r")

        # Collect batch feature
        start_idx = bidx * batch_size
        end_idx = min(start_idx + batch_size, last_batch_idx)
        batch = np.array([feature[idx:idx+timestep] for idx in range(start_idx, end_idx)])  # noqa: E226
        batch = np.expand_dims(batch, axis=3)

        # Predict contour
        batch_pred = model.predict(batch)
        batch_pred = 1 / (1 + np.exp(-expit(batch_pred)))

        # Add the batch results to the output container.
        for idx, pred in enumerate(batch_pred):
            slice_start = start_idx + idx
            slice_end = slice_start + timestep
            output[slice_start:slice_end] += pred
    output = output[pad_len:-pad_len, pad_bottom:-pad_top, 1]  # Remove padding

    # Filter values
    avg_max_val = np.mean(np.max(output, axis=1))
    output = np.where(output > avg_max_val, output, 0)

    # Generate final output F0
    f0 = []  # pylint: disable=invalid-name
    for pitches in output:
        if np.sum(pitches) > 0:
            pidx = np.argmax(pitches)
            f0.append(midi_to_hz(pidx / 4 + LOWEST_MIDI_NOTE))
        else:
            f0.append(0)

    return np.array(f0)
