import numpy as np


def inference(pred, mapping, zzz, cenf, threshold=0.5, max_method="posterior"):
    """Infers pitch contour from the model prediction.

    Parameters
    ----------
    pred:
        The predicted results of the model.
    mapping: 2D numpy array
        The original frequency and time index of patches.
        See ``omnizart.feature.cfp.extract_patch_cfp`` for more details.
    zzz: 2D numpy array
        The original CFP feature.
    cenf: list[float]
        Center frequencies in Hz of each frequency index.
    threshold: float
        Threshold for filtering value of predictions.
    max_method: {'posterior', 'prior'}
        The approach for determine the frequency. Method of *posterior* assigns the
        frequency value according to the given ``mapping`` parameter, and *prior*
        uses the given ``zzz`` feature for the determination.

    Returns
    -------
    contour: 1D numpy array
        Sequence of freqeuncies in Hz, representing the inferred pitch contour.
    """
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
            else:
                raise ValueError(f"Invalid maximum method: {max_method}")
            freq_idx = int(freq_idx)
            contour[int(candidate[freq_idx, 1])] = candidate[freq_idx, 0]

    for idx, cont in enumerate(contour):
        if cont > 1:
            contour[idx] = cenf[int(cont)]

    return contour
