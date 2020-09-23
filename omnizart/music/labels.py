import numpy as np

from omnizart.constants.midi import MUSICNET_INSTRUMENT_PROGRAMS


class LabelType:
    """Defines different types of `music` label.

    Defines functions that converts the customized label format into numpy
    array. With the customized format, it is more flexible to transform
    labels into different different numpy formats according to the usage
    scenario, and also saves a lot of storage space by using the customized
    format.

    Parameters
    ----------
    mode: ['note', 'note-stream', 'pop-note-stream', 'frame', 'frame-stream']
        Mode of label conversion.

        * note: outputs onset and duration channel
        * note-stream: outputs onset and duration channel of instruments (for MusicNet)
        * pop-note-stream: similar to ``note-stream`` mode, but is for ``Pop`` dataset
        * frame: same as ``note`` mode. To truely output duration channel only, use \
        `true-frame` mode.
        * frame-stream: same as ``note-stream``. To truely output duration channel only \
        for each instrument, use ``true-frame-stream`` mode.
    """
    def __init__(self, mode):
        self.mode = mode

        self._classical_channel_mapping = self._init_classical_channel_mapping()
        self._pop_channel_mapping = self._init_pop_channel_mapping()

        self.mode_mapping = {
            "true-frame": {"conversion_func": self.get_frame, "out_classes": 2},
            "frame": {"conversion_func": self.get_frame_onset, "out_classes": 3},
            "note": {"conversion_func": self.get_frame_onset, "out_classes": 3},
            "true-frame-stream": {"conversion_func": self.multi_inst_frm, "out_classes": 12},
            "frame-stream": {"conversion_func": self.multi_inst_note, "out_classes": 23},
            "note-stream": {"conversion_func": self.multi_inst_note, "out_classes": 23},
            "pop-note-stream": {"conversion_func": self.multi_pop_note, "out_classes": 13}
        }
        if mode not in self.mode_mapping:
            raise ValueError(f"Available mode: {self.mode_mapping.keys()}. Provided: {mode}")

    def _init_classical_channel_mapping(self):  # pylint: disable=R0201
        return {val: idx + 1 for idx, val in enumerate(MUSICNET_INSTRUMENT_PROGRAMS)}

    def _init_pop_channel_mapping(self):  # pylint: disable=R0201
        guitar = {i: 1 for i in range(24, 32)}
        bass = {i: 2 for i in range(32, 40)}
        strings = {i: 3 for i in range(40, 56)}
        organ = {i: 4 for i in range(16, 24)}
        piano = {i: 5 for i in range(8)}
        others = {i: 6 for i in range(128)}
        return {**others, **guitar, **bass, **strings, **organ, **piano}

    def get_available_modes(self):
        return list(self.mode_mapping.keys())

    def get_conversion_func(self):
        return self.mode_mapping[self.mode]["conversion_func"]

    def get_out_classes(self):
        return self.mode_mapping[self.mode]["out_classes"]

    def get_frame(self, label):
        return label_conversion(label, channel_mapping=self._classical_channel_mapping, mpe=True)

    def get_frame_onset(self, label):
        frame = self.get_frame(label)
        onset = label_conversion(
            label, channel_mapping=self._classical_channel_mapping, onsets=True, mpe=True
        )[:, :, 1]

        frame[:, :, 1] -= onset
        frm_on = np.dstack([frame, onset])
        frm_on[:, :, 0] = 1 - np.sum(frm_on[:, :, 1:], axis=2)

        return frm_on

    def multi_inst_frm(self, label):
        return label_conversion(label, channel_mapping=self._classical_channel_mapping)

    def multi_inst_note(self, label):
        onsets = label_conversion(label, channel_mapping=self._classical_channel_mapping, onsets=True)
        dura = label_conversion(label, channel_mapping=self._classical_channel_mapping) - onsets
        out = np.zeros(onsets.shape[:-1] + (23,))

        for i in range(len(set(self._classical_channel_mapping.values()))):
            out[:, :, i*2+2] = onsets[:, :, i+1]  # noqa: E226
            out[:, :, i*2+1] = dura[:, :, i+1]  # noqa: E226
        out[:, :, 0] = 1 - np.sum(out[:, :, 1:], axis=2)

        return out

    def multi_pop_note(self, label):
        onsets = label_conversion(
            label, onsets=True, channel_mapping=self._pop_channel_mapping
        )
        dura = label_conversion(
            label, channel_mapping=self._pop_channel_mapping
        ) - onsets
        out = np.zeros(onsets.shape[:-1] + (13,))

        for i in range(len(set(self._pop_channel_mapping.values()))):
            out[:, :, i*2+2] = onsets[:, :, i+1]  # noqa: E226
            out[:, :, i*2+1] = dura[:, :, i+1]  # noqa: E226
        out[:, :, 0] = 1 - np.sum(out[:, :, 1:], axis=2)

        return out


def label_conversion(
    label,
    ori_feature_size=352,
    feature_num=352,
    base=88,
    mpe=False,
    onsets=False,
    channel_mapping=None
):
    assert(ori_feature_size % base == 0)
    scale = ori_feature_size // base

    if channel_mapping is None:
        channel_mapping = {i: i for i in range(1, 129)}

    inst_num = len(set(channel_mapping.keys()))
    output = np.zeros((len(label), ori_feature_size, inst_num+1))  # noqa: E226
    for t, lab in enumerate(label):
        if len(lab) == 0:
            continue

        for pitch, insts in lab.items():
            for inst, prob in insts.items():
                # TODO: Remove minus one in future!!
                inst = int(inst) - 1
                if inst not in channel_mapping:
                    continue

                pitch = int(pitch)
                channel = channel_mapping[inst]
                feat_range = range(pitch*scale, (pitch+1)*scale)  # noqa: E226
                output[t, feat_range, channel] = prob[0]

    if not onsets:
        output = np.where(output > 0, 1, 0)

    pad_b = (feature_num - output.shape[1]) // 2
    pad_t = feature_num - pad_b - output.shape[1]
    b_shape = (len(output), pad_b, output.shape[2])
    t_shape = (len(output), pad_t, output.shape[2])
    bottom = np.zeros(b_shape)
    top = np.zeros(t_shape)
    output = np.concatenate([bottom, output, top], axis=1)

    if mpe:
        mpe_label = np.nanmax(output[:, :, 1:], axis=2)
        output = np.dstack([output[:, :, 0], mpe_label])

    output[:, :, 0] = 1 - np.sum(output[:, :, 1:], axis=2)
    return output
