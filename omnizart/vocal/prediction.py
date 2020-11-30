import numpy as np

from omnizart.utils import get_logger


logger = get_logger("Vocal Predict")


def create_batches(feature, ctx_len=9, batch_size=64):
    feat_pad = np.pad(feature, ((ctx_len, ctx_len), (0, 0), (0, 0)))

    slices = [feat_pad[idx - ctx_len:idx + ctx_len + 1] for idx in range(ctx_len, len(feat_pad) - ctx_len)]
    pad_size = batch_size - len(slices) % batch_size
    payload = np.zeros_like(slices[0])
    for _ in range(pad_size):
        slices.append(payload)
    slices = np.array(slices)
    assert len(slices) % batch_size == 0

    batches = [slices[idx:idx + batch_size] for idx in range(0, len(slices), batch_size)]
    return np.array(batches, dtype=np.float32), pad_size


def predict(feature, model, ctx_len=9, batch_size=16):
    assert feature.shape[1:] == (174, 9)
    batches, pad_size = create_batches(feature, ctx_len=ctx_len, batch_size=batch_size)
    batch_pred = []
    for idx, batch in enumerate(batches):
        print(f"Progress: {idx+1}/{len(batches)}", end="\r")
        batch_pred.append(model.predict(batch))
    pred = np.concatenate(batch_pred)
    return pred[:-pad_size]
