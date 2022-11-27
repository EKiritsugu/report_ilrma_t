# 对于输出的信号进行评估

# 我们仅需要保留如下参数：算法 通道数 si参数

from metrics import si_bss_eval
import os
import soundfile as sf
from nara_wpe.utils import stft, istft
from toolbox import projection_back
import numpy as np
import pandas as pd
import numpy as np
######################################################################################################################

# sou_char to list
def sou2list(a):
    import re
    a = a.replace('\\\\', '\\')
    c = re.findall("WSJ.*?wav", a)
    return c

def str2array(a):
    # print(a)
    a = a.strip('[]')
    # print(a)
    b = a.split(' ')
    # print(b)
    while '' in b:
        b.remove('')
    # if ' ' in b:
    #     b.remove(' ')
    # print(b)
    c = np.array([float(k)
                  for k in b])
    return c

def get_ori_source(source_path):
    import librosa
    # src0, _ = librosa.load(source_path[0], sr=16000)

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
    return sig_np


###################################################################################################################
n_sources = 2
mixed_sig_path = 'mixed/'+str(n_sources) + 'ch/'
save_path = 'wped/'+str(n_sources) + 'ch/'

file_list = os.listdir(mixed_sig_path)

stft_options = dict(size=1024, shift=1024//4)
sampling_rate = 16000

iterations = 100

algorithm = 'ilrma-t-IP'

# 读取excel文件
excel_path = 'mixed/ch_'+str(n_sources)+'.xlsx'
df = pd.read_excel(excel_path)
n_samples = df.shape[0]

n_s = 10# for n_s in range(n_samples):
for n_s in range(100):
    sig_int = df.loc[n_s][0]
    sig_path = 'seped/'+str(n_sources)+'ch/'+algorithm+'/'+str(sig_int)+'.wav'
    mixed_path = 'mixed/'+str(n_sources)+'ch/'+str(sig_int)+'.wav'
    seped_sig = sf.read(sig_path)[0].T
    mixed_sig = sf.read(mixed_path)[0].T

    sou_char = df.loc[n_s][1]
    sou_list = sou2list(sou_char)

    sir_char = df.loc[n_s][2]
    sir_list = str2array(sir_char)

    sdr_char = df.loc[n_s][3]
    sdr_list = str2array(sdr_char)

    sar_char = df.loc[n_s][4]
    sar_list = str2array(sar_char)

    perm_char = df.loc[n_s][5]
    perm_list = str2array(perm_char)
    perm0 = perm_list.astype(int)

    sou_ori = get_ori_source(sou_list)

    sou_ori1 = np.hstack((sou_ori, np.zeros((n_sources , seped_sig.shape[1] - sou_ori.shape[1]))))
    sou_ori2 = np.hstack((sou_ori, np.zeros((n_sources, mixed_sig.shape[1] - sou_ori.shape[1]))))
    sdr1 , sir1 , sar1, perm = si_bss_eval(sou_ori1.T, seped_sig.T)

    import mir_eval

    sdr2, sir2, sar2, perm2 = mir_eval.separation.bss_eval_sources(sou_ori1, seped_sig)
    print(np.mean(sdr2))
    s = np.array([1,1000])
    sdr3, sir3, sar3, perm3 = mir_eval.separation.bss_eval_sources(sou_ori1, s[:,None]* seped_sig)

    from torchmetrics import ScaleInvariantSignalDistortionRatio
    import torch
    sou_ori = sou_ori[::-1, :].copy()
    si_sdr = ScaleInvariantSignalDistortionRatio()
    si0 = si_sdr(torch.from_numpy(mixed_sig), torch.from_numpy(sou_ori2))
    si= si_sdr( torch.from_numpy(s[:,None]*seped_sig), torch.from_numpy( sou_ori1))

    dsdr = si - si0
    print(dsdr)

    # print(sou_list)
    # print(sir_list)
    # print(df.shape)


print('done')



