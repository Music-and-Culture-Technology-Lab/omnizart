import math
import numpy as np

from omnizart.utils import get_logger


logger = get_logger("Beat Prediction")

#: Step size for slicing the feature. Ratio to the timesteps of the model input feature.
STEP_SIZE_RATIO = 0.5


def create_batches(feature, timesteps, batch_size=8):
    """Create a 4D output from the 2D feature for model prediciton.

    Create overlapped input features, and collect feature slices into batches.
    The overlap size is 1/4 length to the timesteps.

    Parameters
    ----------
    feature: 2D numpy array
        The feature representation for the model.
    timesteps: int
        Size of the input feature dimension.
    batch_size: int
        Batch size.

    Returns
    -------
    batches: 4D numpy array
        Batched feature slices with dimension: batches x batch_size x timesteps x feat.
    """
    step_size = round(timesteps * STEP_SIZE_RATIO)
    pad_sides = step_size // 2
    feature = np.pad(feature, ((pad_sides, pad_sides), (0, 0)))

    num_batches = math.ceil(len(feature) / step_size)
    pad_len = num_batches * step_size - len(feature)

    num_pad_batch = batch_size - num_batches % batch_size
    pad_len += num_pad_batch * step_size + step_size
    new_feat = np.pad(feature, ((0, pad_len), (0, 0)))

    batch_feat = [new_feat[idx:idx + timesteps] for idx in range(0, len(new_feat) - step_size, step_size)]
    batches = [batch_feat[idx:idx + batch_size] for idx in range(0, len(batch_feat), batch_size)]
    return np.array(batches)


def merge_batches(batch_pred):
    """Merge the batched predictions back to the 2D output."""
    pred = np.concatenate(batch_pred)
    batches, timesteps, feat_dim = pred.shape

    step_size = round(timesteps * STEP_SIZE_RATIO)
    pad_size = step_size // 2
    out = np.zeros(((batches * timesteps) // 2, feat_dim))
    for idx, pred_slice in enumerate(pred):
        start = idx * step_size
        out[start:start + step_size] = pred_slice[pad_size:-pad_size]
    return out


def predict(feature, model, timesteps=1000, batch_size=64):
    """Predict on the given feature with the model.

    Parameters
    ----------
    feature: 2D numpy array
        Input feature of the model.
    model:
        The pre-trained Tensorflow model.
    timesteps: int
        Size of the input feature dimension.
    batch_size: int
        Batch size for the model input.

    Returns
    -------
    pred: 2D numpy array
        The predicted probabilities of beat and down beat positions.
    """
    logger.debug("Creating batches")
    ori_len = len(feature)
    batches = create_batches(feature, timesteps=timesteps, batch_size=batch_size)

    batch_pred = []
    for idx, batch in enumerate(batches):
        print(f"{idx+1}/{len(batches)}", end="\r")
        batch_pred.append(model.predict(batch))

    logger.debug("Merging batch prediction")
    pred = np.concatenate(batch_pred)  # batches x timesteps x feat
    pred = np.concatenate(pred)  # length x feat
    assert len(pred.shape) == 2
    return pred[:ori_len]
