import numpy as np
from options import *


def normalize(data, mode=opt.norm_mode):
    # Normalize waveforms in each batch
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        max_data[max_data == 0] = 1
        data /= max_data

    elif mode == 'std':
        std_data = np.std(data, axis=0, keepdims=True)
        std_data[std_data == 0] = 1
        data /= std_data
    return data
