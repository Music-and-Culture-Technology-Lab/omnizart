import numpy as np
from scipy.special import expit

from omnizart.models.utils import padding


def generation_prog(model, score_48, time_index, timesteps, batch_size):

    feature_48 = score_48[time_index:time_index + batch_size, :, :]
    feature_48 = np.reshape(feature_48, (batch_size, timesteps, 384, 1))

    input_features = {'input_score_48': feature_48}

    probas = model.predict(input_features, batch_size=batch_size)

    return probas


def inference(feature, model, timestep=128, batch_size=10, feature_num=384):

    f_48_p, p_t, p_b = padding(feature, feature_num, timestep, dimension=True)
    f_48_s = np.zeros((len(f_48_p), timestep, feature_num))

    for i in range(len(f_48_s) - timestep):
        f_48_s[i] = f_48_p[i:i + timestep]

    extract_result_seg = np.zeros(f_48_s.shape + (2,))
    extract_result_seg_flatten = np.zeros(f_48_p.shape + (2,))

    iter_num = int(np.ceil(((len(f_48_s) - timestep) / batch_size)))

    for i in range(1, iter_num + 1):
        print("batch: {}/{}".format(i, iter_num), end="\r")
        time_index = i * batch_size
        probs = generation_prog(
            model, f_48_s,
            time_index=time_index,
            timesteps=timestep,
            batch_size=batch_size
        )
        probs = 1 / (1 + np.exp(-expit(probs)))
        extract_result_seg[time_index:time_index + batch_size] = probs

    for i in range(len(f_48_s) - timestep):
        extract_result_seg_flatten[i:i + timestep] += extract_result_seg[i]

    extract_result_seg = extract_result_seg_flatten[timestep:-timestep, p_t:-p_b, 1]
    avg = 0

    for i, step in enumerate(extract_result_seg):
        maximum = np.sort(step)[-1]
        avg += maximum
        extract_result_seg[i][extract_result_seg[i] < maximum] = 0

    avg /= extract_result_seg.shape[0]
    extract_result_seg[extract_result_seg < avg] = 0
    extract_result_seg[extract_result_seg > avg] = 1

    return extract_result_seg
