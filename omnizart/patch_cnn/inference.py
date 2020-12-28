import numpy as np


def inference(pred, mapping, zzz, cenf, threshold=0.5, max_method="posterior"):
    pred = pred[:, 1]

    pred_idx = np.where(pred > threshold)
    probs = np.expand_dims(pred[pred_idx[0]], axis=-1)
    maps = mapping[pred_idx[0]]
    maps = np.concatenate([maps, probs], axis=1)
    maps = maps[maps[:, 1].argsort()]

    contour = np.zeros(int(np.max(maps)) + 1)
    for tidx in range(len(probs)):
        candidate = maps[np.where(maps[:, 1] == tidx)[0]]
        if len(candidate) < 1:
            continue
        if len(candidate) == 1:
            contour[int(candidate[0, 1])] = candidate[0, 0]
        else:
            if max_method == "posterior":
                freq_idx = np.where(candidate[:, 2] == np.max(candidate[:, 2]))[0]
            elif max_method == "prior":
                freq_idx = zzz[candidate[:, 0].astype('int'), tidx].argmax(axis=0)
            freq_idx = int(freq_idx)
            contour[int(candidate[freq_idx, 1])] = candidate[freq_idx, 0]

    for idx, cont in enumerate(contour):
        if cont > 1:
            contour[idx] = cenf[int(cont)]

    return contour
