import math
import numpy as np

from omnizart.utils import get_logger


logger = get_logger("Beat Prediction")


def create_batches(feature, timesteps, batch_size=8):
    """Create a 3D output from a 2D input for model prediciton.
    """
    num_batches = math.ceil(len(feature) / timesteps)
    pad_len = num_batches * timesteps - len(feature)

    num_pad_batch = batch_size - num_batches % batch_size
    pad_len += num_pad_batch * timesteps
    new_feat = np.pad(feature, ((0, pad_len), (0, 0)))

    batch_feat = [new_feat[idx:idx + timesteps] for idx in range(0, len(new_feat), timesteps)]
    batches = [batch_feat[idx:idx + batch_size] for idx in range(0, len(batch_feat), batch_size)]
    return np.array(batches)


def predict(feature, model, timesteps=1000, batch_size=64):
    logger.debug("Creating batches")
    batches = create_batches(feature, timesteps=timesteps, batch_size=batch_size)

    batch_pred = []
    for idx, batch in enumerate(batches):
        print(f"{idx+1}/{len(batches)}", end="\r")
        batch_pred.append(model.predict(batch))

    logger.debug("Merging batch prediction")
    pred = np.concatenate(batch_pred)  # batch_size x timesteps x feat
    pred = np.concatenate(pred)  # length x feat
    assert len(pred.shape) == 2
    return pred[:len(feature)]
