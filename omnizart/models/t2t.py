"""Implementation of memory efficient attention.

Original implemetation are from `tensor2tensor <https://github.com/tensorflow/tensor2tensor>`_.
Rewrite in tensorflow 2.0.
"""

# pylint: disable=C0103,R0914,E1120

import functools
import operator

import tensorflow as tf


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def reshape_range(tensor, i, j, shape):
    """Reshapes a tensor between dimensions i and j."""
    t_shape = shape_list(tensor)
    target_shape = t_shape[:i] + shape + t_shape[j:]
    return tf.reshape(tensor, target_shape)


def cast_like(x, y):
    """Cast x to y's dtype, if necessary."""
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    if x.dtype.base_dtype == y.dtype.base_dtype:
        return x

    cast_x = tf.cast(x, y.dtype)
    if cast_x.device != x.device:
        x_name = "(eager Tensor)"
        try:
            x_name = x.name
        except AttributeError:
            pass
        tf.compat.v1.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name, x.device, cast_x.device)
    return cast_x


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
  The first of these two dimensions is n.
  Args:
    x: a Tensor with shape [..., m]
    n: an integer.
  Returns:
    a Tensor with shape [..., n, m/n]
  """
    x_shape = shape_list(x)
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])


def split_heads_2d(x, num_heads):
    """Split channels (dimension 3) into multiple heads (becomes dimension 1).
  Args:
    x: a Tensor with shape [batch, height, width, channels]
    num_heads: an integer
  Returns:
    a Tensor with shape [batch, num_heads, height, width, channels / num_heads]
  """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 3, 1, 2, 4])


def pad_to_multiple_2d(x, block_shape):
    """Making sure x is a multiple of shape.
  Args:
    x: a [batch, heads, h, w, depth] or [batch, h, w, depth] tensor
    block_shape: a 2-d list of integer shapes
  Returns:
    padded_x: a [batch, heads, h, w, depth] or [batch, h, w, depth] tensor
  """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    if len(old_shape) == 4:
        height_padding = -shape_list(x)[1] % block_shape[0]
        width_padding = -shape_list(x)[2] % block_shape[1]
        paddings = [[0, 0], [0, height_padding], [0, width_padding], [0, 0]]
    elif len(old_shape) == 5:
        height_padding = -shape_list(x)[2] % block_shape[0]
        width_padding = -shape_list(x)[3] % block_shape[1]
        paddings = [[0, 0], [0, 0], [0, height_padding], [0, width_padding], [0, 0]]

    padded_x = tf.pad(x, paddings)
    padded_shape = padded_x.get_shape().as_list()
    padded_shape = padded_shape[:-1] + [last]
    padded_x.set_shape(padded_shape)
    return padded_x


def gather_indices_2d(x, block_shape, block_stride):
    """Getting gather indices."""
    # making an identity matrix kernel
    kernel = tf.eye(block_shape[0] * block_shape[1])
    kernel = reshape_range(kernel, 0, 1, [block_shape[0], block_shape[1], 1])
    # making indices [1, h, w, 1] to appy convs
    x_shape = shape_list(x)
    indices = tf.range(x_shape[2] * x_shape[3])
    indices = tf.reshape(indices, [1, x_shape[2], x_shape[3], 1])
    indices = tf.nn.conv2d(
        tf.cast(indices, tf.float32), kernel, strides=[1, block_stride[0], block_stride[1], 1], padding="VALID"
    )
    # making indices [num_blocks, dim] to gather
    dims = shape_list(indices)[:3]
    if all([isinstance(dim, int) for dim in dims]):
        num_blocks = functools.reduce(operator.mul, dims, 1)
    else:
        num_blocks = tf.reduce_prod(dims)
    indices = tf.reshape(indices, [num_blocks, -1])
    return tf.cast(indices, tf.int32)


def gather_blocks_2d(x, indices):
    """Gathers flattened blocks from x."""
    x_shape = shape_list(x)
    x = reshape_range(x, 2, 4, [tf.reduce_prod(x_shape[2:4])])
    # [length, batch, heads, dim]
    x_t = tf.transpose(x, [2, 0, 1, 3])
    x_new = tf.gather(x_t, indices)
    # returns [batch, heads, num_blocks, block_length ** 2, dim]
    return tf.transpose(x_new, [2, 3, 0, 1, 4])


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
  Args:
    x: a Tensor with shape [..., a, b]
  Returns:
    a Tensor with shape [..., ab]
  """
    x_shape = shape_list(x)
    a, b = x_shape[-2:]
    return tf.reshape(x, x_shape[:-2] + [a*b])  # noqa: E226


def combine_heads_2d(x):
    """Inverse of split_heads_2d.
  Args:
    x: a Tensor with shape
      [batch, num_heads, height, width, channels / num_heads]
  Returns:
    a Tensor with shape [batch, height, width, channels]
  """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 3, 1, 4]))


def embedding_to_padding(emb):
    """Calculates the padding mask based on which embeddings are all zero.
  We have hacked symbol_modality to return all-zero embeddings for padding.
  Args:
    emb: a Tensor with shape [..., depth].
  Returns:
    a float Tensor with shape [...]. Each element is 1 if its corresponding
    embedding vector is all zero, and is 0 otherwise.
  """
    emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
    return tf.compat.v1.to_float(tf.equal(emb_sum, 0.0))


def scatter_blocks_2d(x, indices, shape):
    """scatters blocks from x into shape with indices."""
    x_shape = shape_list(x)
    # [length, batch, heads, dim]
    x_t = tf.transpose(tf.reshape(x, [x_shape[0], x_shape[1], -1, x_shape[-1]]), [2, 0, 1, 3])
    x_t_shape = shape_list(x_t)
    indices = tf.reshape(indices, [-1, 1])
    scattered_x = tf.scatter_nd(indices, x_t, x_t_shape)
    scattered_x = tf.transpose(scattered_x, [1, 2, 0, 3])
    return tf.reshape(scattered_x, shape)


def mixed_precision_is_enabled(activation_dtype=None, weight_dtype=None, hparams=None):
    assert not (
        hparams and (activation_dtype or weight_dtype)
    ), "Provide only hparams or activation_dtype and weight_dtype"
    if hparams and hasattr(hparams, "activation_dtype") and hasattr(hparams, "weight_dtype"):
        activation_dtype = hparams.activation_dtype
        weight_dtype = hparams.weight_dtype
    return activation_dtype == tf.float16 and weight_dtype == tf.float32


def maybe_upcast(logits, activation_dtype=None, weight_dtype=None, hparams=None):
    if mixed_precision_is_enabled(activation_dtype, weight_dtype, hparams):
        return tf.cast(logits, tf.float32)
    return logits


def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
    """Like tf.nn.dropout but takes broadcast_dims instead of noise_shape.
  Instead of specifying noise_shape, this function takes broadcast_dims -
  a list of dimension numbers in which noise_shape should be 1.  The random
  keep/drop tensor has dimensionality 1 along these dimensions.
  Args:
    x: a floating point tensor.
    keep_prob: A scalar Tensor with the same type as x.
      The probability that each element is kept.
    broadcast_dims: an optional list of integers
      the dimensions along which to broadcast the keep/drop flags.
    **kwargs: keyword arguments to tf.nn.dropout other than "noise_shape".
  Returns:
    Tensor of the same shape as x.
  """
    assert "noise_shape" not in kwargs
    if broadcast_dims:
        shape = tf.shape(x)
        ndims = len(x.get_shape())
        # Allow dimensions like "-1" as well.
        broadcast_dims = [dim + ndims if dim < 0 else dim for dim in broadcast_dims]
        kwargs["noise_shape"] = [1 if i in broadcast_dims else shape[i] for i in range(ndims)]
    return tf.compat.v1.nn.dropout(x, keep_prob, **kwargs)


def dot_product_attention(
    q,
    k,
    v,
    bias,
    dropout_rate=0.0,
    name=None,
    save_weights_to=None,
    dropout_broadcast_dims=None,
    activation_dtype=None,
    weight_dtype=None,
):
    """Dot-product attention.

    Parameters
    ----------
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims: an optional list of integers less than rank of q.
      Specifies in which dimensions to broadcast the dropout decisions.
    activation_dtype: Used to define function activation dtype when using
      mixed precision.
    weight_dtype: The dtype weights are stored in when using mixed precision
    hard_attention_k: integer, if > 0 triggers hard attention (picking top-k)
      hard_attention_k <= 0.

  Returns
  -------
    Tensor with shape [..., length_q, depth_v].
  """
    with tf.compat.v1.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]) as scope:
        logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
        if bias is not None:
            bias = cast_like(bias, logits)
            logits += bias
        # If logits are fp16, upcast before softmax
        logits = maybe_upcast(logits, activation_dtype, weight_dtype)
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights = cast_like(weights, q)
        if save_weights_to is not None:
            save_weights_to[scope.name] = weights
            save_weights_to[scope.name + "/logits"] = logits
        # Drop out attention links for each head.
        weights = dropout_with_broadcast_dims(weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
        return tf.matmul(weights, v)


def local_attention_2d(q, k, v, query_shape=(8, 16), memory_flange=(8, 16), name=None):
    """Strided block local self-attention.
  The 2-D sequence is divided into 2-D blocks of shape query_shape. Attention
  for a given query position can only see memory positions less than or equal to
  the query position. The memory positions are the corresponding block with
  memory_flange many positions to add to the height and width of the block
  (namely, left, top, and right).
  Args:
    q: a Tensor with shape [batch, heads, h, w, depth_k]
    k: a Tensor with shape [batch, heads, h, w, depth_k]
    v: a Tensor with shape [batch, heads, h, w, depth_v]. In the current
      implementation, depth_v must be equal to depth_k.
    query_shape: an tuple indicating the height and width of each query block.
    memory_flange: an integer indicating how much to look in height and width
      from each query block.
    name: an optional string
  Returns:
    a Tensor of shape [batch, heads, h, w, depth_v]
  """
    with tf.compat.v1.variable_scope(name, default_name="local_self_attention_2d", values=[q, k, v]):
        v_shape = shape_list(v)

        # Pad query, key, value to ensure multiple of corresponding lengths.
        q = pad_to_multiple_2d(q, query_shape)
        k = pad_to_multiple_2d(k, query_shape)
        v = pad_to_multiple_2d(v, query_shape)
        paddings = [[0, 0], [0, 0], [memory_flange[0], memory_flange[1]], [memory_flange[0], memory_flange[1]], [0, 0]]
        k = tf.pad(k, paddings)
        v = tf.pad(v, paddings)

        # Set up query blocks.
        q_indices = gather_indices_2d(q, query_shape, query_shape)
        q_new = gather_blocks_2d(q, q_indices)

        # Set up key and value blocks.
        memory_shape = (query_shape[0] + 2 * memory_flange[0], query_shape[1] + 2 * memory_flange[1])
        k_and_v_indices = gather_indices_2d(k, memory_shape, query_shape)
        k_new = gather_blocks_2d(k, k_and_v_indices)
        v_new = gather_blocks_2d(v, k_and_v_indices)

        attention_bias = tf.expand_dims(tf.compat.v1.to_float(embedding_to_padding(k_new)) * -1e9, axis=-2)
        output = dot_product_attention(q_new, k_new, v_new, attention_bias, dropout_rate=0.0, name="local_2d")
        # Put representations back into original shapes.
        padded_q_shape = shape_list(q)
        output = scatter_blocks_2d(output, q_indices, padded_q_shape)

        # Remove the padding if introduced.
        output = tf.slice(output, [0, 0, 0, 0, 0], [-1, -1, v_shape[2], v_shape[3], -1])
        return output
