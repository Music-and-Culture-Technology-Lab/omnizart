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


def merge_batches(batch_pred):
    assert len(batch_pred.shape) == 4

    batches, batch_size, frm_len, out_classes = batch_pred.shape
    total_len = batches * batch_size + frm_len - 1
    output = np.zeros((total_len, out_classes))
    for bidx, batch in enumerate(batch_pred):
        for fidx, frame in enumerate(batch):
            start_idx = bidx * batch_size + fidx
            output[start_idx:start_idx + frm_len] += frame

    max_len = min(frm_len - 1, len(output) - frm_len)
    output[max_len:-max_len] /= max_len + 1
    for idx in range(max_len):
        output[idx] /= idx + 1
        output[-1 - idx] /= idx + 1
    return output


def predict(feature, model, ctx_len=9, batch_size=16):
    assert feature.shape[1:] == (174, 9)
    batches, pad_size = create_batches(feature, ctx_len=ctx_len, batch_size=batch_size)
    batch_pred = []
    for idx, batch in enumerate(batches):
        print(f"Progress: {idx+1}/{len(batches)}", end="\r")
        batch_pred.append(model.predict(batch))
    pred = merge_batches(np.array(batch_pred))
    return pred[ctx_len:-pad_size - ctx_len]
