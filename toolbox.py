# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:08:09 2022

@author: AR
"""

import numpy as np
# from pyroomacoustics.bss import projection_back
import soundfile as sf
from scipy import signal
import librosa

def load_resampled_audio(fileway , fs):
    '''
    This function accepts path-like object and samplying frequency. The returned value is an array of dowmsampled waveform.
    '''
    tmp ,sr = sf.read(fileway)
    _,nch = np.shape(tmp)
    for i in range(nch):
        tmp2 = librosa.resample(tmp[:,i] , sr ,fs)
        L_tmp2 = len(tmp2)
        tmp[:L_tmp2,i] = tmp2
    Sig_out = tmp[:L_tmp2,:]
    return Sig_out

def tensor_H(T):
    return np.conj(T).swapaxes(1, 2)

def projection_back(Y, ref, clip_up=None, clip_down=None):
    """
    Parameters
    ----------
    Y: array_like (n_frames, n_bins, n_channels)
        The STFT data to project back on the reference signal
    ref: array_like (n_frames, n_bins)
        The reference signal
    clip_up: float, optional
        Limits the maximum value of the gain (default no limit)
    clip_down: float, optional
        Limits the minimum value of the gain (default no limit)
    """

    num = np.sum(np.conj(ref[:, :, None]) * Y, axis=0)
    denom = np.sum(np.abs(Y) ** 2, axis=0)

    c = np.ones(num.shape, dtype=np.complex)
    I = denom > 0.0
    c[I] = num[I] / denom[I]

    if clip_up is not None:
        I = np.logical_and(np.abs(c) > clip_up, np.abs(c) > 0)
        c[I] *= clip_up / np.abs(c[I])

    if clip_down is not None:
        I = np.logical_and(np.abs(c) < clip_down, np.abs(c) > 0)
        c[I] *= clip_down / np.abs(c[I])
    return c

def Mstft(Sig_ori , fft_len=2048 , lap_len=1024):
    # Sig_ori: t , nch
    # return : n_freq n_frame n_ch
    [_,M] = np.shape(Sig_ori)

    ##stft
    _,_,Zxx0 = signal.stft(Sig_ori[:,0] , nperseg=fft_len , noverlap=lap_len)
    a,b = np.shape(Zxx0)
    Sw = np.zeros((a,b,M) , dtype=complex)
    Sw[:,:,0] = Zxx0
    for i in range(1,M):
        f_list,_,Zxx = signal.stft(Sig_ori[:,i] , nperseg=fft_len , noverlap=lap_len)
        Sw[:,:,i] = Zxx

    return Sw

def Mistft(Sw , fft_len= 2048  ,lap_len = 1024):
    # Sw : n_freq n_frame n_ch
    # return: t , nch
    K = Sw.shape[2]
    ## istft
    _ , tmp = signal.istft(Sw[:,:,0], nperseg=fft_len , noverlap=lap_len)
    St_hat = np.zeros((len(tmp) , K))
    St_hat[:,0] = np.real(tmp)
    for i in range(1,K):
        _ , tmp = signal.istft(Sw[:,:,i], nperseg=fft_len , noverlap=lap_len)
        St_hat[:,i] = np.real(tmp)
    return St_hat

