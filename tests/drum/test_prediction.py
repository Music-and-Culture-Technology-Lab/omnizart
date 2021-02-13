import pytest
import numpy as np

from omnizart.drum import prediction as putils


@pytest.mark.parametrize("shape,mini_beat_per_seg,batch_size", [
    ((20, 13, 4), 6, 4),
    ((10, 11, 12), 5, 12),
    ((51, 33, 7), 11, 5),
    ((2, 30, 4), 7, 10)
])
def test_create_batches_size(shape, mini_beat_per_seg, batch_size):
    data = np.arange(np.prod(shape)).reshape(shape)

    target_first_dim = max(1, int(np.ceil((shape[0]-mini_beat_per_seg+1)/batch_size)))
    target_shape = (target_first_dim, batch_size, shape[1], shape[2], mini_beat_per_seg)

    result, pad_size = putils.create_batches(data, mini_beat_per_seg, b_size=batch_size)
    assert result.shape == target_shape


def test_create_batches_value():
    data = np.arange(24).reshape((2, 3, 4))
    expected = np.array([[
        [[[ 0., 12.,  0.], [ 1., 13.,  0.], [ 2., 14.,  0.], [ 3., 15.,  0.]],
         [[ 4., 16.,  0.], [ 5., 17.,  0.], [ 6., 18.,  0.], [ 7., 19.,  0.]],
         [[ 8., 20.,  0.], [ 9., 21.,  0.], [10., 22.,  0.], [11., 23.,  0.]]],

        [[[ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.]],
         [[ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.]],
         [[ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.]]],

        [[[ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.]],
         [[ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.]],
         [[ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.]]],
    ]])
    result, pad_size = putils.create_batches(data, 3, 3)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("shape", [
    (1, 3, 3, 4, 1), (3, 4, 5, 6, 1), (10, 6, 12, 15, 1)
])
def test_merge_batches_shape(shape):
    mini_beat_per_seg = shape[3]
    out_classes = shape[2]
    batch_size = shape[1]
    batches = shape[0]

    target_first_dim = batch_size*batches + mini_beat_per_seg - 1
    target_shape = (target_first_dim, out_classes)
    data = np.arange(np.prod(shape)).reshape(shape)

    result = putils.merge_batches(data)
    assert result.shape == target_shape


@pytest.mark.parametrize("shape,value", [
    ((1, 4, 8, 12, 1), 1),
    ((2, 5, 9, 3, 1), 2.5),
    ((6, 5, 4, 10, 1), 3),
    ((1, 9, 33, 2, 1), 5)
])
def test_merge_batches_value(shape, value):
    data = np.zeros(shape)
    data.fill(value)
    result = putils.merge_batches(data)
    assert np.array_equiv(result, value)
