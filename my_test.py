# 我真的受不了这个傻逼了，我决定我自己写个程序来测一下
# 反正还是经典三声源吧
###### 抽取混合音频
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
from metrics import si_bss_eval
import os
import soundfile as sf
from nara_wpe.utils import stft, istft
from toolbox import projection_back
import numpy as np

n_ch = 2




class room_para:
    def __init__(self, n_sources):

        self.n_ch = n_sources
        self.T60 = 10 * random.randint(20, 60)
        self.fs = 16000

        self.wall =np.array([8, 5])
        self.upper = 4


        self.z_source = 1.5 + 0.5 * np.random.rand(1, n_sources).squeeze()


        self.xyz_source = np.array([
            # [4.5-0.1, 4 , 1.8],
                                    [4.5-2 , 2+1.4, 1.7],
                                    [4.5+2, 2+1.4 , 2]])
        self.xyz_mic = np.array([
            # [4.5, 2, 1.5],
                                [4.5 - 0.2, 2, 1.5],
                                [4.5 + 0.2, 2, 1.5]])
        self.xyz_box = np.array([8, 5, 4])


    def pic_show(self):
        fig,ax = plt.subplots()

        rect = mpathes.Rectangle([0,0], self.wall[0], self.wall[1], color = 'pink')
        ax.add_patch(rect)

        x_mic = self.loc_mic[:,0]
        y_mic = self.loc_mic[:,1]
        plt.scatter(x_mic, y_mic, c='blue', label = 'function')

        plt.scatter(self.loc_source[:,0], self.loc_source[:,1], c='green', label = 'function')


        plt.axis('equal')
        plt.grid()
        plt.show()

def simulate(n_sources = 2,draw = False):

    box = room_para(n_sources)
    # print(box.xyz_source)

    rt60 = box.T60 / 1000
    fs = box.fs
    n_ch = box.n_ch

    e_absorption , max_order = pra.inverse_sabine(rt60, box.xyz_box)
    echo_room = pra.ShoeBox(box.xyz_box, materials = pra.Material(e_absorption), max_order = max_order, fs = fs)

    echo_room.add_microphone_array(box.xyz_mic.T)
    for i in range(n_ch):
        echo_room.add_source(box.xyz_source[i,:].tolist())


    echo_room.compute_rir()

    if draw:
        fig, ax = echo_room.plot()
        ax.set_xlim([-1, 10])
        ax.set_ylim([-1, 10])
        ax.set_zlim([-1, 5])
        fig.show()


    ret = echo_room.rir

    len_ret = np.array([len(ret[i][j])
                        for i in range(n_sources)
                        for j in range(n_sources)])
    lr = np.min(len_ret)
    for i in range(n_sources):
        for j in range(n_sources):
            ret[i][j] = ret[i][j][:lr]

    ret2 = np.array(ret , dtype = float)

    return ret2


def get_sources(n_sources = 2):


    dataset_dir = r'WSJ_ilrma-t'

    speakers = os.listdir(dataset_dir)
    n_speaker = len(speakers)
    speaker_set = [os.listdir(os.path.join(dataset_dir, i))
        for i in speakers]

    speaker_value = random.sample(np.arange(n_speaker).tolist(), n_sources)

    data = pd.DataFrame(speaker_set, speakers)
    # print(data)

    source_choosed = [
        data.loc[speakers[ i ]][random.randint(0,len(speaker_set[i])-1)]
        for i in speaker_value
        ]

    source_path = [
        os.path.join(dataset_dir,speakers[speaker_value[i]] , source_choosed[i])
                   for i in range(n_sources)
    ]

    src0, _ = librosa.load(source_path[0], sr=16000)

    sig = []
    sig_len = []
    for i in range(n_sources):
        src, _ = librosa.load(source_path[i], sr=16000)
        sig.append(src)
        sig_len.append(src.shape[0])
    # print(max(sig_len))aa
    sig_np = np.zeros((n_sources, max(sig_len)))
    for i in range(n_sources):
        sig_np[i, : sig_len[i]] = sig[i]
    # print(np.shape(sig_np))
    return sig_np, source_path


def get_mixed_sig(n_sources = 2, an = True):
    def get_noise(n_sources = 2, len_n = 160000):
        dataset_dir = r'noise_wav'

        speakers = os.listdir(dataset_dir)


        sample_dir = random.sample(speakers, n_sources)
        print(sample_dir)
        sample_path = [
            os.path.join(dataset_dir, sample_dir[i])
                        for i in range(n_sources)
        ]

        sig = []
        sig_len = []
        for i in range(n_sources):
            src, _ = librosa.load(sample_path[i], sr=16000)
            sig.append(src)
            sig_len.append(src.shape[0])
        # print(max(sig_len))
        sig_np = np.zeros((n_sources, max(sig_len)))
        for i in range(n_sources):
            sig_np[i, : sig_len[i]] = sig[i]###其实这一大段可以精简一些，不过我觉得不管它了


        start = random.randint(0, max(sig_len) - len_n)
        sig_out = sig_np[:, start:start + len_n]
        return sig_out

    def add_noise(mixed_sig ):

        n_sources = mixed_sig.shape[0]
        len_n = mixed_sig.shape[1]

        noise = get_noise(n_sources, len_n)

        noise = noise / np.var(noise)

        SNR = random.randint(10,30)
        var = np.sqrt(n_sources * 10**(-SNR) )

        noise_sig_add = var * noise

        return noise_sig_add + mixed_sig

    rir = simulate(n_sources = n_sources)
    sig, src_path = get_sources(n_sources = n_sources)

    sig_mixed = []
    for i in range(n_sources):
        a = convolve(sig[0] , rir[0][i])
        for j in range(1 , n_sources):
            a = a + convolve(sig[j] , rir[j][i])
        sig_mixed.append(a)


    if an != True:
        sig_mixed = np.array(sig_mixed)/np.var(sig_mixed[0])
        sig_mixed *= 1/np.max(np.abs(sig_mixed), axis = 1 )[:,None]
        return sig_mixed, sig ,src_path
    else:
        sig_mixed_n = add_noise(np.array(sig_mixed))
        # sf.write('mix_test.wav', sig_mixed[1], 16000)
        return sig_mixed_n, sig ,src_path


mixed_wav, ori_sig, src_path = get_mixed_sig(n_ch)
sf.write('test_wav/MT_ori.wav', mixed_wav.T , 16000)

ori_sig = np.hstack((ori_sig, np.zeros((n_ch , mixed_wav.shape[1] - ori_sig.shape[1]))))
si_sdr0 , si_sir0 , si_sar0, si_perm = si_bss_eval(ori_sig.T, mixed_wav.T)


from torchmetrics import ScaleInvariantSignalDistortionRatio
import torch
si_sdr_f = ScaleInvariantSignalDistortionRatio()
si0 = si_sdr_f(torch.from_numpy(mixed_wav), torch.from_numpy(ori_sig))
print(si0)

from mir_eval.separation import bss_eval_sources
sdr0 , sir0 , sar0, perm = bss_eval_sources(ori_sig, mixed_wav)


#####处理
stft_options = dict(size=1024, shift=1024//4)
sampling_rate = 16000
delay = 2
iterations = 100
taps = 5
n_iter = 100
n_components = 2



y = mixed_wav
Y = stft(y, **stft_options)
X = Y.transpose(2, 0, 1).copy()
del Y, y# 把可能混淆的变量删除
#######################################################################################################################
# input&output X: n_freq, n_ch, n_frame

X = X.transpose(2, 0, 1)
algorithm = 'ilrma-t-iss-seq'
algorithm_list = ['ilrma-t-IP', 'ilrma-t-iss-seq', 'ilrma-t-iss-joint', 'ilrma-IP', 'auxiva']
######################################################################################################################
from ilrma_t_function import *
import pyroomacoustics as pra
for algorithm in algorithm_list:
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

    sf.write('test_wav/MT_'+algorithm+'.wav', z.T , 16000)






##### 评估
print('done')