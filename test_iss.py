# 这个文件旨在生成混合后的数据，并且保存至对应的文件夹中
# 诸多文件夹路径中，注意有dataset——dir noise——dir

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

class room_para:
    def __init__(self, n_sources):
        def mic_judge():
            center_room = self.wall / 2
            xy_delta = self.center_mic - center_room
            distance = np.sqrt(np.sum(xy_delta**2))
            if distance < 0.2:
                return False
            else:
                return True

        def source_judge():
            dis_mic = np.array([ np.sqrt(np.sum((self.loc_source[i] - self.center_mic)**2))
                       for i in range(n_sources)
                    ])
            dis_center = np.array([ np.sqrt(np.sum((self.loc_source[i] - self.wall / 2)**2))
                       for i in range(n_sources)
                    ])
            dis_source = np.array([ np.sqrt(np.sum((self.loc_source[i] - self.loc_source[j] )**2))
                           for i in range(n_sources)
                           for j in range(i+1, n_sources)])

            if np.min(dis_mic) > 1.5 and np.min(dis_center) > 0.2 and np.min(dis_source) > 1 :
                return True

        self.n_ch = n_sources
        self.T60 = 10 * random.randint(20, 60)
        self.fs = 16000

        self.wall = 5 + 5* np.array([random.random() , random.random()])
        self.upper = 3 + random.random()


        self.r_mic = 0.075 + 0.05 * random.random()
        self.z_mic = 1 + random.random()
        while True:
            self.center_mic = self.r_mic +  np.array([
                (self.wall[0] - 2* self.r_mic) * random.random() ,
                (self.wall[1] - 2* self.r_mic) * random.random()])
            if mic_judge():
                break
        self.theta_mic = (1 / n_sources) * np.arange(0 , n_sources) * 360 + random.randint(0, 360)
        self.loc_mic = self.r_mic * np.array( [
            np.cos(self.theta_mic * 2* np.pi / 360) ,
            np.sin(self.theta_mic * 2* np.pi / 360)]).T \
                       + self.center_mic

        self.z_source = 1.5 + 0.5 * np.random.rand(1, n_sources).squeeze()
        while True:
            self.loc_source = self.wall * np.random.rand(n_sources , 2)
            if source_judge():
                break

        self.xyz_source = np.hstack((self.loc_source, self.z_source[:, None]))
        self.xyz_mic = (np.hstack((self.loc_source, self.z_mic* np.ones((n_sources,1)))))
        self.xyz_box = np.hstack((self.wall, self.upper))


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
    # print(max(sig_len))
    sig_np = np.zeros((n_sources, max(sig_len)))
    for i in range(n_sources):
        sig_np[i, : sig_len[i]] = sig[i]
    # print(np.shape(sig_np))
    return sig_np, source_path


def get_mixed_sig(n_sources = 2, SNR = None):

    def add_noise(mixed_sig , SNR = 10):

        n_sources = mixed_sig.shape[0]
        len_f = mixed_sig.shape[1]

        pw_n = 10**(- SNR / 10) * n_sources

        noise_path = r'noise.wav'
        noise, _ = librosa.load(noise_path, sr=16000)
        noise = noise / np.var(noise)

        n_frames = noise.shape[0] // mixed_sig.shape[1]
        noise_frame = np.array([noise[i*len_f : (i+1)*len_f ]
            for i in range(n_frames)
        ])
        # noise_cs = random.sample(np.arange(n_frames).tolist(), n_sources)
        noise_cs = [1,1]
        noise_sig_add = pw_n * noise_frame[noise_cs, :]

        return noise_sig_add + mixed_sig

    rir = simulate(n_sources = n_sources)
    sig, src_path = get_sources(n_sources = n_sources)

    sig_mixed = []
    for i in range(n_sources):
        a = convolve(sig[0] , rir[i][0])
        for j in range(1 , n_sources):
            a = a + convolve(sig[j] , rir[i][j])
        sig_mixed.append(a)


    if SNR is None:
        sig_mixed = np.array(sig_mixed)/np.var(sig_mixed[0])
        sig_mixed *= 1/np.max(np.abs(sig_mixed), axis = 1 )[:,None]
        return sig_mixed, sig ,src_path
    else:
        sig_mixed_n = add_noise(sig_mixed, SNR)
        # sf.write('mix_test.wav', sig_mixed[1], 16000)
        return sig_mixed_n, sig ,src_path



n_samples = 333
n_ch = 3
L_frame = 1024
hop = L_frame//4
win_a = pra.hann(L_frame)
win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

print('working')


mixed_wav, ori_sig, src_path = get_mixed_sig(n_ch)

ori_sig = np.hstack((ori_sig, np.zeros((n_ch , mixed_wav.shape[1] - ori_sig.shape[1]))))
sdr0 , sir0 , sar0, perm = si_bss_eval(ori_sig.T, mixed_wav.T)

print(np.shape(ori_sig))

from ilrma_iss import ilrma_iss, auxiva_iss
X = pra.transform.stft.analysis(mixed_wav.T, L_frame,  hop, win=win_a)
Y = ilrma_iss(X, n_iter=30, proj_back=True)
y = pra.transform.stft.synthesis(Y, L_frame, hop, win=win_s)
print(np.shape(y))
sf.write('septest0.wav', y[:,0], 16000)
sf.write('septest1.wav', y[:,1], 16000)










