import numpy as np

from omnizart.constants.feature import DOWN_SAMPLE_TO_SAPMLING_RATE, HOP_SIZE


def get_frame_by_time(time_sec):
    return int(round(time_sec * DOWN_SAMPLE_TO_SAPMLING_RATE / HOP_SIZE))
