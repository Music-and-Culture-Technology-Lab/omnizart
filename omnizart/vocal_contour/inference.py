import numpy as np
import tqdm

from omnizart.models.utils import note_res_downsampling, padding


def sigmoid(tensor):
    return 1 / (1 + np.exp(-tensor))


def generation_prog(model, score_48, score_12, time_index, timesteps, batch_size):

    feature_48 = score_48[time_index:time_index + batch_size, :, :]
    feature_48 = np.reshape(feature_48, (batch_size, timesteps, 384, 1))

    feature_12 = score_12[time_index:time_index + batch_size, :, :]
    feature_12 = np.reshape(feature_12, (batch_size, timesteps, 128, 1))

    input_features = {'input_score_48': feature_48, 'input_score_12': feature_12}

    probas = model.predict(input_features, batch_size=batch_size)

    return probas


def inference(feature, model, timestep=128, batch_size=10, feature_num_12=128, feature_num_48=384, channel=1):

    f_12 = note_res_downsampling(feature)
    f_12_p = padding(f_12, feature_num_12, timestep)
    f_12_s = np.zeros((f_12_p.shape[0], timestep, f_12_p.shape[1]))

    f_48_p, p_t, p_b = padding(feature, feature_num_48, timestep, dimension=True)
    f_48_s = np.zeros((f_48_p.shape[0], timestep, f_48_p.shape[1]))

    for i in range(len(f_12_s) - timestep):
        f_12_s[i] = f_12_p[i:i + timestep]
        f_48_s[i] = f_48_p[i:i + timestep]

    extract_result_seg = np.zeros(f_48_s.shape + (2,))
    extract_result_seg_flatten = np.zeros(f_48_p.shape + (2,))

    iter_num = int(np.ceil(((len(f_12_s) - timestep) / batch_size)))

    for i in tqdm.tqdm(range(iter_num)):
        time_index = i * batch_size
        probs = generation_prog(
            model, f_48_s, f_12_s,
            time_index=time_index,
            timesteps=timestep,
            batch_size=batch_size
        )
        probs = sigmoid(probs)
        extract_result_seg[time_index:time_index + batch_size] = probs

    for i in range(len(f_12_s) - timestep):
        extract_result_seg_flatten[i:i + timestep] += extract_result_seg[i]

    extract_result_seg = extract_result_seg_flatten[timestep:-timestep, p_t:-p_b, 1]
    avg = 0
    if channel == 2:
        extract_result_seg_unvoiced = extract_result_seg_flatten[timestep:-timestep, p_t:-p_b, 0]
        avg_unvoiced = np.sum(np.sum(extract_result_seg_unvoiced)) / extract_result_seg_unvoiced.size

        extract_result_seg_unvoiced[extract_result_seg_unvoiced > avg_unvoiced] = 200
        extract_result_seg_unvoiced[extract_result_seg_unvoiced < avg_unvoiced] = 1
        extract_result_seg_unvoiced[extract_result_seg_unvoiced == 200] = 0
        extract_result_seg = extract_result_seg * extract_result_seg_unvoiced

    for i, step in enumerate(extract_result_seg):
        maximum = np.sort(step)[-1]
        avg += maximum
        extract_result_seg[i][extract_result_seg[i] < maximum] = 0

    avg /= extract_result_seg.shape[0]
    extract_result_seg[extract_result_seg < avg] = 0
    extract_result_seg[extract_result_seg > avg] = 1

    return extract_result_seg
