# 用于测试封装后函数的代码



import os
import soundfile as sf
from nara_wpe.utils import stft, istft
from toolbox import projection_back
import numpy as np
######################################################################################################################



algorithm = 'ilrma-IP'
for n_s in range(3):

    n_sources = n_s + 2
    mixed_sig_path = 'mixed/'+str(n_sources) + 'ch/'
    # save_path = 'wped/'+str(n_sources) + 'ch/'
    save_path = 'seped/'+str(n_sources)+'ch/'+algorithm+'/'
    # os.mkdir(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_list = os.listdir(mixed_sig_path)

    stft_options = dict(size=1024, shift=1024//4)
    sampling_rate = 16000
    delay = 2
    iterations = 100
    taps = 5


    for wav_name in file_list:
        print(wav_name)
        wav_save = 'seped/'+str(n_sources)+'ch/'+algorithm+'/'+wav_name
        # print(wav_save)

        y = sf.read(mixed_sig_path+wav_name)[0]
        y = y.T
        Y = stft(y, **stft_options)
        X = Y.transpose(2, 0, 1).copy()
        del Y, y# 把可能混淆的变量删除
        #######################################################################################################################
        # input&output X: n_freq, n_ch, n_frame

        X = X.transpose(2, 0, 1)
        n_iter = 100
        n_components = 2

        ######################################################################################################################
        from ilrma_t_function import *
        import pyroomacoustics as pra
        if algorithm == 'ilrma-t-IP':
            Y = ilrma_t_iss_joint(X, n_iter = n_iter)
        elif algorithm == 'ilrma-t-iss-seq':
            Y = ilrma_t_iss_seq(X, n_iter = n_iter)
        elif algorithm == 'ilrma-t-iss-joint':
            Y = ilrma_t_iss_joint(X, n_iter = n_iter)
        elif algorithm == 'ilrma-IP':
            Y = pra.bss.ilrma(X, n_iter = n_iter)
        elif algorithm == 'auxiva':
            Y = pra.bss.auxiva(X, n_iter = n_iter)


        Z = Y.transpose(1, 2, 0)

        #######################################################################################################################
        z = istft(Z.transpose(1, 2, 0), size=stft_options['size'], shift=stft_options['shift'])
        for i in range(n_sources):
            sf.write(wav_save, z.T , 16000)

    print('done!')