# 这个来测试一下，换用CMU的声源来测试一下

# 
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


class room_para_r:
    def __init__(self, n_sources):
        def mic_judge():
            center_room = self.wall / 2
            xy_delta = self.center_mic - center_room
            distance = np.sqrt(np.sum(xy_delta ** 2))
            if distance < 0.2:
                return False
            else:
                return True

        def source_judge():
            dis_mic = np.array([np.sqrt(np.sum((self.loc_source[i] - self.center_mic) ** 2))
                                for i in range(n_sources)
                                ])
            dis_center = np.array([np.sqrt(np.sum((self.loc_source[i] - self.wall / 2) ** 2))
                                   for i in range(n_sources)
                                   ])
            dis_source = np.array([np.sqrt(np.sum((self.loc_source[i] - self.loc_source[j]) ** 2))
                                   for i in range(n_sources)
                                   for j in range(i + 1, n_sources)])

            if np.min(dis_mic) > 1.5 and np.min(dis_center) > 0.2 and np.min(dis_source) > 1:
                return True

        self.n_ch = n_sources
        self.T60 = 10 * random.randint(20, 60)
        self.fs = 16000

        self.wall = 5 + 5 * np.array([random.random(), random.random()])
        self.upper = 3 + random.random()

        self.r_mic = 0.075 + 0.05 * random.random()
        self.z_mic = 1 + random.random()
        while True:
            self.center_mic = self.r_mic + np.array([
                (self.wall[0] - 2 * self.r_mic) * random.random(),
                (self.wall[1] - 2 * self.r_mic) * random.random()])
            if mic_judge():
                break
        self.theta_mic = (1 / n_sources) * np.arange(0, n_sources) * 360 + random.randint(0, 360)
        self.loc_mic = self.r_mic * np.array([
            np.cos(self.theta_mic * 2 * np.pi / 360),
            np.sin(self.theta_mic * 2 * np.pi / 360)]).T \
                       + self.center_mic

        self.z_source = 1.5 + 0.5 * np.random.rand(1, n_sources).squeeze()
        while True:
            self.loc_source = self.wall * np.random.rand(n_sources, 2)
            if source_judge():
                break

        self.xyz_source = np.hstack((self.loc_source, self.z_source[:, None]))
        self.xyz_mic = (np.hstack((self.loc_source, self.z_mic * np.ones((n_sources, 1)))))
        self.xyz_box = np.hstack((self.wall, self.upper))

    def pic_show(self):
        fig, ax = plt.subplots()

        rect = mpathes.Rectangle([0, 0], self.wall[0], self.wall[1], color='pink')
        ax.add_patch(rect)

        x_mic = self.loc_mic[:, 0]
        y_mic = self.loc_mic[:, 1]
        plt.scatter(x_mic, y_mic, c='blue', label='function')

        plt.scatter(self.loc_source[:, 0], self.loc_source[:, 1], c='green', label='function')

        plt.axis('equal')
        plt.grid()
        plt.show()


class room_para:
    def __init__(self, n_sources):
        self.n_ch = n_sources
        self.T60 = 600
        self.fs = 16000

        self.wall = np.array([8, 5])
        self.upper = 4

        self.z_source = 1.5 + 0.5 * np.random.rand(1, n_sources).squeeze()

        self.xyz_source = np.array([
            # [4.5-0.1, 4 , 1.8],
            [4.5 - 1, 2 + 1.7, 1.7],
            [4.5 + 1, 2 + 1.7, 2]])
        self.xyz_mic = np.array([
            # [4.5, 2, 1.5],
            [4.5 - 0.05, 2, 1.5],
            [4.5 + 0.05, 2, 1.5]])
        self.xyz_box = np.array([8, 5, 4])

    def pic_show(self):
        fig, ax = plt.subplots()

        rect = mpathes.Rectangle([0, 0], self.wall[0], self.wall[1], color='pink')
        ax.add_patch(rect)

        x_mic = self.loc_mic[:, 0]
        y_mic = self.loc_mic[:, 1]
        plt.scatter(x_mic, y_mic, c='blue', label='function')

        plt.scatter(self.loc_source[:, 0], self.loc_source[:, 1], c='green', label='function')

        plt.axis('equal')
        plt.grid()
        plt.show()


def simulate(n_sources=2, draw=False):
    box = room_para_r(n_sources)
    # print(box.xyz_source)

    rt60 = box.T60 / 1000
    fs = box.fs
    n_ch = box.n_ch

    e_absorption, max_order = pra.inverse_sabine(rt60, box.xyz_box)
    echo_room = pra.ShoeBox(box.xyz_box, materials=pra.Material(e_absorption), max_order=max_order, fs=fs)

    echo_room.add_microphone_array(box.xyz_mic.T)
    for i in range(n_ch):
        echo_room.add_source(box.xyz_source[i, :].tolist())

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

    ret2 = np.array(ret, dtype=float)

    return ret2


def get_noise(n_sources=2, len_n=160000):
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
        sig_np[i, : sig_len[i]] = sig[i]  ###其实这一大段可以精简一些，不过我觉得不管它了

    start = random.randint(0, max(sig_len) - len_n)
    sig_out = sig_np[:, start:start + len_n]
    return sig_out


def get_sources_2(n_sources=2, T = 10):
    dataset_dir = r'wsj'

    speakers = os.listdir(dataset_dir)

    sample_dir = random.sample(speakers, n_sources)
    print(sample_dir)
    sample_path = [
        os.path.join(dataset_dir, sample_dir[i])
        for i in range(n_sources)
    ]

    sig = []

    for i in range(n_sources):
        src, _ = librosa.load(sample_path[i], sr=16000)
        sig_len = src.shape[0]
        start = random.randint(0, sig_len - T * 16000)
        sig.append(src[start:start+T * 16000])



    return np.array(sig)

def get_sources_3(n_sources=2, T = 10):
    dataset_dir = r'wsj'

    speakers = os.listdir(dataset_dir)

    sample_dir = random.sample(speakers, n_sources)
    print(sample_dir)
    sample_path = [
        os.path.join(dataset_dir, sample_dir[i])
        for i in range(n_sources)
    ]

    sig = []

    for i in range(n_sources):
        src, _ = librosa.load(sample_path[i], sr=16000)
        sig_len = src.shape[0]
        start = random.randint(0, sig_len - T * 16000)
        sig.append(src[start:start+T * 16000])



    return sig


def get_sources(n_sources=2):
    dataset_dir = r'CMU'

    speakers = os.listdir(dataset_dir)
    n_speaker = len(speakers)
    speaker_set = [os.listdir(os.path.join(dataset_dir, i))
                   for i in speakers]

    speaker_value = random.sample(np.arange(n_speaker).tolist(), n_sources)

    data = pd.DataFrame(speaker_set, speakers)
    # print(data)

    source_choosed = [
        data.loc[speakers[i]][random.randint(0, len(speaker_set[i]) - 1)]
        for i in speaker_value
    ]

    source_path = [
        os.path.join(dataset_dir, speakers[speaker_value[i]], source_choosed[i])
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
    return sig_np




def add_noise(mixed_sig):
    n_sources = mixed_sig.shape[0]
    len_n = mixed_sig.shape[1]

    noise = get_noise(n_sources, len_n)

    noise = noise / np.var(noise)

    SNR = random.randint(10, 30)
    var = np.sqrt(n_sources * 10 ** (-SNR))

    noise_sig_add = var * noise

    return noise_sig_add + mixed_sig


def get_mixed_sig(n_sources=2, an=True):
    def get_noise(n_sources=2, len_n=160000):
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
            sig_np[i, : sig_len[i]] = sig[i]  ###其实这一大段可以精简一些，不过我觉得不管它了

        start = random.randint(0, max(sig_len) - len_n)
        sig_out = sig_np[:, start:start + len_n]
        return sig_out

    def add_noise(mixed_sig):

        n_sources = mixed_sig.shape[0]
        len_n = mixed_sig.shape[1]

        noise = get_noise(n_sources, len_n)

        noise = noise / np.var(noise)

        SNR = random.randint(10, 30)
        var = np.sqrt(n_sources * 10 ** (-SNR))

        noise_sig_add = var * noise

        return noise_sig_add + mixed_sig

    rir = simulate(n_sources=n_sources)
    sig, src_path = get_sources(n_sources=n_sources)

    sig_mixed = []
    for i in range(n_sources):
        a = convolve(sig[0], rir[0][i])
        for j in range(1, n_sources):
            a = a + convolve(sig[j], rir[j][i])
        sig_mixed.append(a)

    if an != True:
        sig_mixed = np.array(sig_mixed) / np.var(sig_mixed[0])
        sig_mixed *= 1 / np.max(np.abs(sig_mixed), axis=1)[:, None]
        return sig_mixed, sig, src_path
    else:
        sig_mixed_n = add_noise(np.array(sig_mixed))
        # sf.write('mix_test.wav', sig_mixed[1], 16000)
        return sig_mixed_n, sig, src_path


def simulate_2(n_ch=2):
    box = room_para_r(n_ch)

    rt60 = box.T60 / 1000
    fs = box.fs

    e_absorption, max_order = pra.inverse_sabine(rt60, box.xyz_box)
    room = pra.ShoeBox(box.xyz_box, materials=pra.Material(e_absorption), max_order=max_order, fs=fs)
    # R = pra.linear_2D_array(center_xy, n_mic, 0, dis_mic)
    room.add_microphone_array(box.xyz_mic.T)

    sig_np = get_sources_2(n_ch)

    for i in range(n_ch):
        room.add_source(box.xyz_source[i], signal=sig_np[i], delay=0)

    room.simulate()

    Sig_ori = room.mic_array.signals
    Sig_ori = Sig_ori[:, :sig_np.shape[1]]

    return Sig_ori, sig_np

def simulate_3(n_ch = 2, sinr = 600):

    sig = get_sources(n_ch)
    sig_np = np.array(sig)
    from room_builder import random_room_builder
    Sig_ori, sig_np = random_room_builder(sig , n_ch, sinr = sinr)

    sig_np = sig_np[:n_ch , :]
    Sig_ori = Sig_ori / np.max(np.abs(Sig_ori), axis = 1)[:, None]
    sig_np = sig_np / np.max(np.abs(sig_np), axis = 1)[:, None]
    

    return Sig_ori, sig_np

print('working')
n_ch = 2
for i in range(333):
    print(i)

    mixed_wav, ref_sig = simulate_3(n_ch)
    # mixed_wav = add_noise(mixed_wav)
    sf.write('test_wav/MT_ori.wav', mixed_wav.T, 16000)

    # ori_sig = np.hstack((ori_sig, np.zeros((n_ch, mixed_wav.shape[1] - ori_sig.shape[1]))))
    si_sdr0, si_sir0, si_sar0, si_perm = si_bss_eval(ref_sig.T, mixed_wav.T)


    from mir_eval.separation import bss_eval_sources

    sdr0, sir0, sar0, perm = bss_eval_sources(ref_sig, mixed_wav)

    #####处理
    stft_options = dict(size=1024, shift=1024 // 4)
    sampling_rate = 16000
    delay = 2
    iterations = 100
    taps = 20
    n_iter = 100
    n_components = 2

    y = mixed_wav
    sf.write('test_wav/mixed.wav', y.T, 16000)
    Y = stft(y, **stft_options)
    X = Y.transpose(2, 0, 1).copy()
    del Y, y  # 把可能混淆的变量删除
    #######################################################################################################################
    # input&output X: n_freq, n_ch, n_frame

    X = X.transpose(2, 0, 1)
    algorithm = 'ilrma-t-iss-seq'
    # algorithm_list = ['my_ilrma_t', 'wpe', 'wpe+ilrma-IP', 'ilrma-t-IP', 'ilrma-t-iss-seq', 'ilrma-t-iss-joint',
    #                   'ilrma-IP', 'ilrma-iss', 'auxiva']
    algorithm_list = [ 'my5_ilrma_t','my4_ilrma_t','my3_ilrma_t','my2_ilrma_t','wpe_6','wpe+ilrma-IP',  'ilrma-t-iss-seq',
                      'ilrma-IP', 'auxiva']
    # algorithm_list = ['wpe+ilrma-IP',  'ilrma-t-iss-seq',
                    #   'ilrma-IP', 'auxiva']
    ######################################################################################################################
    from ilrma_t_function import *
    import pyroomacoustics as pra

    for algorithm in algorithm_list:
        print(algorithm)
        if algorithm == 'ilrma-t-IP':
            Y = ilrma_t_iss_joint(X, n_iter=n_iter)
        elif algorithm == 'ilrma-t-iss-seq':
            Y = ilrma_t_iss_seq(X, n_iter=n_iter)
        elif algorithm == 'ilrma-t-iss-joint':
            Y = ilrma_t_iss_joint(X, n_iter=n_iter)
        elif algorithm == 'ilrma-IP':
            Y = pra.bss.ilrma(X, n_iter=n_iter)
        elif algorithm == 'ilrma-iss':
            from ilrma_iss import ilrma_iss

            Y = ilrma_iss(X, n_iter=n_iter)
        elif algorithm == 'auxiva':
            Y = pra.bss.auxiva(X, n_iter=n_iter)
        elif algorithm == 'wpe+ilrma-IP':
            from nara_wpe.wpe import wpe

            Y = wpe(X.transpose(1, 2, 0),
                    taps=taps,
                    delay=delay,
                    iterations=50,
                    statistics_mode='full'
                    ).transpose(2, 0, 1)
            Y = pra.bss.ilrma(Y, n_iter=50)
        elif algorithm == 'wpe_6':


            Y = wpe_v(X,
                    taps=taps,
                    delay=delay,
                    iterations=iterations,
                    )

        elif algorithm == 'my_ilrma_t':
            Y = my_ilrma_t(X, n_iter=n_iter)
        elif algorithm == 'my2_ilrma_t':
            Y = my2_ilrma_t(X, n_iter=n_iter)
        elif algorithm == 'my3_ilrma_t':
            Y = my3_ilrma_t(X, n_iter=n_iter)
        elif algorithm == 'my4_ilrma_t':
            Y = my4_ilrma_t(X, n_iter=n_iter)
        elif algorithm == 'my5_ilrma_t':
            Y = my5_ilrma_t(X, n_iter=n_iter)

        Z = Y.transpose(1, 2, 0)

        #######################################################################################################################
        z = istft(Z.transpose(1, 2, 0), size=stft_options['size'], shift=stft_options['shift'])
        z = z[:, :ref_sig.shape[1]]

        sf.write('test_wav/MT_2_' + algorithm + '.wav', z.T, 16000)
        ##### 评估
        

        si_sdr1, si_sir1, si_sar1, si_perm1 = si_bss_eval(ref_sig.T, z.T)
        dsd = si_sdr1[si_perm1] - si_sdr0[si_perm]
        dsi = si_sir1[si_perm1] - si_sir0[si_perm]
        dsa = si_sar1[si_perm1] - si_sar0[si_perm]


        from mir_eval.separation import bss_eval_sources

        sdr1, sir1, sar1, perm1 = bss_eval_sources(ref_sig, z)
        dd = sdr1[perm1] - sdr0[perm]
        di = sir1[perm1] - sir0[perm]
        da = sar1[perm1] - sar0[perm]

        file_txt = open('test_wav/cmu_wpe_f_' + str(n_ch) + '.txt', mode='a')
        file_txt.write(algorithm + '\t' + str(np.mean(dd)) +  '\t' + str(np.mean(dsd)) + '\n')
        file_txt.close()

print('done')