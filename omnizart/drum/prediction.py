# pylint: disable=E1101
import numpy as np

from omnizart.utils import get_logger


logger = get_logger("Drum Prediction")


def create_batches(feature, mini_beat_per_seg, b_size=6):
    """Create a 4D input for model prediction.

    Parameters
    ----------
    feature: 3D numpy array
        Should be in shape [mini_beat_pos x time x freq].
    mini_beat_per_seg: int
        Number of mini beats in one segment (a beat).
    b_size: int
        Output batch size.

    Returns
    -------
    batch_feature: 5D numpy array
        Dimensions are [batches x b_size x time x freq x mini_beat_per_seg].
    pad_size: int
        The additional padded size at the end of the batch.
    """
    assert (len(feature.shape) == 3), f"Invalid feature shape: {feature.shape}. Should be three dimensional."

    # Pad zeros to the end of the feature if not long enough.
    if len(feature) < mini_beat_per_seg:
        pad_len = mini_beat_per_seg - len(feature)
        pads = np.zeros((pad_len, *feature.shape[1:]))
        feature = np.concatenate([feature, pads])

    hops = len(feature) - mini_beat_per_seg + 1
    hop_list = []
    for idx in range(hops):
        feat = feature[idx:idx + mini_beat_per_seg]
        hop_list.append(np.transpose(feat, axes=[1, 2, 0]))

    total_batches = int(np.ceil(len(hop_list) / b_size))
    batch_feature = []
    for idx in range(total_batches):
        batch_feature.append(hop_list[idx * b_size:(idx+1) * b_size])  # noqa: E226

    zero_feat = np.zeros_like(hop_list[0])
    pad_size = b_size - len(batch_feature[-1])
    for _ in range(pad_size):
        batch_feature[-1].append(zero_feat)

    return np.array(batch_feature), pad_size


def merge_batches(batch_pred):
    """Reverse process of create_batches.

    Merges a 5D batched-prediction into 2D output.
    """
    assert len(batch_pred.shape) == 5
    assert batch_pred.shape[-1] == 1

    logger.debug("Batch prediction shape: %s", batch_pred.shape)
    batch_pred = np.transpose(batch_pred, axes=[0, 1, 3, 2, 4])
    batches, b_size, mini_beat_per_seg, out_classes = batch_pred.shape[:4]
    pred = np.zeros((batches*b_size + mini_beat_per_seg - 1, out_classes))  # noqa: E226
    for b_idx, batch in enumerate(batch_pred):
        for s_idx, step in enumerate(batch):
            start_idx = b_idx*b_size + s_idx  # noqa: E226
            end_idx = start_idx + mini_beat_per_seg
            pred[start_idx:end_idx] += step.squeeze()

    max_len = min(mini_beat_per_seg - 1, len(pred) - mini_beat_per_seg)
    pred[max_len:-max_len] /= max_len + 1
    for idx in range(max_len):
        pred[idx] /= idx + 1
        pred[-1 - idx] /= idx + 1
    return pred


def predict(patch_cqt_feature, model, mini_beat_per_seg, batch_size=32):
    batches, pad_size = create_batches(patch_cqt_feature, mini_beat_per_seg, b_size=batch_size)
    batch_pred = []
    for idx, batch in enumerate(batches):
        print(f"{idx+1}/{len(batches)}", end="\r")
        batch_pred.append(model.predict(batch))
    pred = merge_batches(np.array(batch_pred))
    return pred[:-pad_size]   # Remove padding
