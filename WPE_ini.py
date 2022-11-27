# 这个文件用于生成采用WPE初始化后的音频信号

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import pyroomacoustics as pra
import os
import pandas as pd
import librosa
from scipy.signal import convolve
import soundfile as sf
from nara_wpe.wpe import wpe
from nara_wpe.wpe import get_power
from nara_wpe.utils import stft, istft, get_stft_center_frequencies

for n_s in range(3):
    n_sources = n_s + 2
    mixed_sig_path = 'mixed/'+str(n_sources) + 'ch/'
    save_path = 'wped/'+str(n_sources) + 'ch/'

    file_list = os.listdir(mixed_sig_path)

    stft_options = dict(size=1024, shift=1024//4)
    sampling_rate = 16000
    delay = 2
    iterations = 100
    taps = 5


    for wav_name in file_list:
        print(wav_name)
        y = sf.read(mixed_sig_path+wav_name)[0]
        y = y.T
        Y = stft(y, **stft_options).transpose(2, 0, 1)
        S = Y.shape[:-2]
        D = Y.shape[-2]

        Z = wpe(
            Y,
            taps=taps,
            delay=delay,
            iterations=iterations,
            statistics_mode='full'
        ).transpose(1, 2, 0)
        z = istft(Z, size=stft_options['size'], shift=stft_options['shift'])
        sf.write(save_path+wav_name, z.T , 16000)


