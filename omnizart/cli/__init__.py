"""Command line interface of `omnizart`
"""


def silence_tensorflow():
    """To silence the warning and error messages from Tensorflow.

    The end users should not be exposed to the huge amount of confusing messages,
    but developers may need to. Therefore this function is defined here for CLI
    only, and not in omnizart API files.
    """
    # pylint: disable=E0401,C0415
    from tensorflow.compat.v1 import logging as tf_logger
    tf_logger.set_verbosity(tf_logger.ERROR)
