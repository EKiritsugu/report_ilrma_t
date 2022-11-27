##这个代码用于测试ILRMA-T-ISS
# seq算法已经成功

import os
import soundfile as sf
from nara_wpe.utils import stft, istft
from toolbox import projection_back
import numpy as np
######################################################################################################################


def segment_axis(
        x,
        length,
        shift,
        axis=-1,
        end='cut',  # in ['pad', 'cut', None]
        pad_mode='constant',
        pad_value=0,
):

    backend = {
        'numpy': 'numpy',
        'cupy.core.core': 'cupy',
        'torch': 'torch',
    }[x.__class__.__module__]

    if backend == 'numpy':
        xp = np
    elif backend == 'cupy':
        import cupy
        xp = cupy
    elif backend == 'torch':
        import torch
        xp = torch
    else:
        raise Exception('Can not happen')

    try:
        ndim = x.ndim
    except AttributeError:
        # For Pytorch 1.2 and below
        ndim = x.dim()

    axis = axis % ndim

    # Implement negative shift with a positive shift and a flip
    # stride_tricks does not work correct with negative stride
    if shift > 0:
        do_flip = False
    elif shift < 0:
        do_flip = True
        shift = abs(shift)
    else:
        raise ValueError(shift)

    if pad_mode == 'constant':
        pad_kwargs = {'constant_values': pad_value}
    else:
        pad_kwargs = {}

    # Pad
    if end == 'pad':
        if x.shape[axis] < length:
            npad = np.zeros([ndim, 2], dtype=int)
            npad[axis, 1] = length - x.shape[axis]
            x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)
        elif shift != 1 and (x.shape[axis] + shift - length) % shift != 0:
            npad = np.zeros([ndim, 2], dtype=int)
            npad[axis, 1] = shift - ((x.shape[axis] + shift - length) % shift)
            x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)

    elif end == 'conv_pad':
        assert shift == 1, shift
        npad = np.zeros([ndim, 2], dtype=int)
        npad[axis, :] = length - shift
        x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)
    elif end is None:
        assert (x.shape[axis] + shift - length) % shift == 0, \
            '{} = x.shape[axis]({}) + shift({}) - length({})) % shift({})' \
            ''.format((x.shape[axis] + shift - length) % shift,
                      x.shape[axis], shift, length, shift)
    elif end == 'cut':
        pass
    else:
        raise ValueError(end)

    # Calculate desired shape and strides
    shape = list(x.shape)
    # assert shape[axis] >= length, shape
    del shape[axis]
    shape.insert(axis, (x.shape[axis] + shift - length) // shift)
    shape.insert(axis + 1, length)

    def get_strides(array):
        try:
            return list(array.strides)
        except AttributeError:
            # fallback for torch
            return list(array.stride())

    strides = get_strides(x)
    strides.insert(axis, shift * strides[axis])

    # Alternative to np.ndarray.__new__
    # I am not sure if np.lib.stride_tricks.as_strided is better.
    # return np.lib.stride_tricks.as_strided(
    #     x, shape=shape, strides=strides)
    try:
        if backend == 'numpy':
            x = np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape)
        elif backend == 'cupy':
            x = x.view()
            x._set_shape_and_strides(strides=strides, shape=shape)
        elif backend == 'torch':
            import torch
            x = torch.as_strided(x, size=shape, stride=strides)
        else:
            raise Exception('Can not happen')

        # return np.ndarray.__new__(np.ndarray, strides=strides,
        #                           shape=shape, buffer=x, dtype=x.dtype)
    except Exception:
        print('strides:', get_strides(x), ' -> ', strides)
        print('shape:', x.shape, ' -> ', shape)
        try:
            print('flags:', x.flags)
        except AttributeError:
            pass  # for pytorch
        print('Parameters:')
        print('shift:', shift, 'Note: negative shift is implemented with a '
                               'following flip')
        print('length:', length, '<- Has to be positive.')
        raise
    if do_flip:
        return xp.flip(x, axis=axis)
    else:
        return x

def build_y_tilde(Y, taps, delay):

    S = Y.shape[:-2]
    D = Y.shape[-2]
    T = Y.shape[-1]

    def pad(x, axis=-1, pad_width=taps + delay - 1):#将信号向后移pad_width帧
        npad = np.zeros([x.ndim, 2], dtype=int)
        npad[axis, 0] = pad_width
        x = np.pad(x,
                   pad_width=npad,
                   mode='constant',
                   constant_values=0)
        return x


    # ToDo: write the shape
    Y_ = pad(Y)
    Y_ = np.moveaxis(Y_, -1, -2)#23两列转置
    Y_ = np.flip(Y_, axis=-1)#交换了转置后的每一组位置
    Y_ = np.ascontiguousarray(Y_)#没啥卵用
    Y_ = np.flip(Y_, axis=-1)#又换了回去
    Y_ = segment_axis(Y_, taps, 1, axis=-2)
    Y_ = np.flip(Y_, axis=-2)
    if delay > 0:
        Y_ = Y_[..., :-delay, :, :]
    Y_ = np.reshape(Y_, list(S) + [T, taps * D])
    Y_ = np.moveaxis(Y_, -2, -1)

    return Y_

def _stable_solve(A, B):

    def get_working_shape(shape):
        import functools
        import operator
        "Flattens all but the last two dimension."
        product = functools.reduce(operator.mul, [1] + list(shape[:-2]))
        return [product] + list(shape[-2:])

    assert A.shape[:-2] == B.shape[:-2], (A.shape, B.shape)
    assert A.shape[-1] == B.shape[-2], (A.shape, B.shape)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        shape_A, shape_B = A.shape, B.shape
        assert shape_A[:-2] == shape_A[:-2]
        working_shape_A = get_working_shape(shape_A)
        working_shape_B = get_working_shape(shape_B)
        A = A.reshape(working_shape_A)
        B = B.reshape(working_shape_B)

        C = np.zeros_like(B)
        for i in range(working_shape_A[0]):
            # lstsq is much slower, use it only when necessary
            try:
                C[i] = np.linalg.solve(A[i], B[i])
            except np.linalg.LinAlgError:
                C[i] = np.linalg.lstsq(A[i], B[i])[0]
        return C.reshape(*shape_B)








######################################################################################################################

n_sources = 4
mixed_sig_path = 'mixed/'+str(n_sources) + 'ch/'
save_path = 'wped/'+str(n_sources) + 'ch/'

file_list = os.listdir(mixed_sig_path)

stft_options = dict(size=1024, shift=1024//4)
sampling_rate = 16000
delay = 2
iterations = 100
taps = 5

wav_name = file_list[100]
y = sf.read(mixed_sig_path+wav_name)[0]
y = y.T
Y = stft(y, **stft_options)
X = Y.transpose(2, 0, 1).copy()
del Y, y# 把可能混淆的变量删除
#######################################################################################################################
# input&output X: n_freq, n_ch, n_frame

X = X.transpose(2, 0, 1)
n_iter = 20
n_components = 2
if True:
    """
    X: ndarray (nframes, nfrequencies, nchannels)
    Returns an (nframes, nfrequencies, nsources) array.
    """
    n_frames, n_freq, n_chan = X.shape

    # default to determined case

    n_src = X.shape[2]

    # Only supports determined case
    assert n_chan == n_src, "There should be as many microphones as sources"

    W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)


    # initialize the nonnegative matrixes with random values
    T = np.array(0.1 + 0.9 * np.random.rand(n_src, n_freq, n_components))
    V = np.array(0.1 + 0.9 * np.random.rand(n_src, n_frames, n_components))
    R = np.zeros((n_src, n_freq, n_frames))

    lambda_aux = np.zeros(n_src)
    eps = 1e-15
    eyes = np.tile(np.eye(n_chan, n_chan), (n_freq, 1, 1))

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_src, n_frames), dtype=X.dtype)
    Y_wpe = np.zeros((n_freq, n_src, n_frames), dtype=X.dtype)
    X_original = X

    X = X.transpose([1, 2, 0]).copy()
    # print(np.shape(X))# n_freq, n_ch, n_frames

    np.matmul(T, V.swapaxes(1, 2), out=R)
    # Compute the demixed output


    # Y2.shape == R.shape == (n_src, n_freq, n_frames)
    Y2 = np.power(abs(Y.transpose([1, 0, 2])), 2.0)
    iR = 1 / R

    ##WPE
    X_line = build_y_tilde(Y, taps, delay)
    X_tilde = np.array([np.vstack((X[i], X_line[i]))
                           for i in range(n_freq)])

    n_tilde = X_tilde.shape[1]
    G = np.array([np.eye(X_tilde.shape[1] , X_tilde.shape[1]) for f in range(n_freq)], dtype=X.dtype)
    P = G[:, :n_src, :]
    W_p = P[:, :, :n_src]

    A_p = np.zeros((n_freq, X_tilde.shape[1]), dtype=X.dtype)
    eyes_t = np.tile(np.eye( X_tilde.shape[1],  X_tilde.shape[1], dtype = X.dtype), (n_freq, 1, 1))
    # print(P[0,:,:])
    # print(np.shape(X_tilde))
    V_nm = np.zeros((n_freq, X_tilde.shape[1], X_tilde.shape[1]), dtype=X.dtype)
    Y_tmp = np.zeros((n_freq, n_tilde, n_frames), dtype = X.dtype)



    def demix_t(Y_wpe,X_tilde,P):
        Y_wpe[:, :, :] = np.matmul(P, X_tilde)
    def demix_tmp(Y_tmp, X_tilde, G):
        Y_tmp[:, :, :] = np.matmul(G, X_tilde)
    demix_t(Y_wpe, X_tilde, P)
    demix_tmp(Y_tmp, X_tilde, G)

    iR_tmp = np.zeros( (n_tilde, n_freq, n_frames), dtype=R.dtype)
    v2 = np.zeros((n_freq, n_tilde), dtype = X.dtype)
    ###########

    for epoch in range(n_iter):
        print(epoch)
        # simple loop as a start
        for s in range(n_src):
            ## NMF
            ######
            T[s, :, :] *= np.sqrt(np.dot(Y2[s, :, :]* iR[s, :, :] ** 2, V[s, :, :])/np.dot(iR[s, :, :], V[s, :, :]))
            T[T < eps] = eps
            R[s, :, :] = np.dot(T[s, :, :], V[s, :, :].T)
            R[R < eps] = eps
            iR[s, :, :] = 1 / R[s, :, :]
            V[s, :, :] *= np.sqrt(np.dot(Y2[s, :, :].T * iR[s, :, :].T ** 2, T[s, :, :])/np.dot(iR[s, :, :].T, T[s, :, :]))
            V[V < eps] = eps
            R[s, :, :] = np.dot(T[s, :, :], V[s, :, :].T)
            R[R < eps] = eps
            iR[s, :, :] = 1 / R[s, :, :]


        ##############
        iR_tmp[:n_src, :, :] = iR[:,:,:]
        for s in range(n_tilde):  # 修改一下，这一行去掉于不去掉区别重大
            v_num = (Y * iR.swapaxes(0, 1)) @ np.conj(Y_tmp[:, s, :, None])  # (n_freq, n_tilde, 1)
            # v_num_2 = v_num[:, :n_src, :]
            # OP: n_frames * n_src
            v_denom = iR.swapaxes(0, 1) @ np.abs(Y_tmp[:, s, :, None]) ** 2   # (n_freq, n_tilde, 1)
            # v_denom_2 = v_denom[:, :n_src, :]
            v_denom[v_denom < eps] = eps
            # OP: n_src
            v = v_num[:, :, 0] / v_denom[:, :, 0]
            # OP: 1
            if s < n_src:
                v[:, s] = 1.0 - np.sqrt(n_frames) / np.sqrt(v_denom[:, s, 0])
            v2[:,:n_src ]= v[:,:]

            # update demixed signals
            # OP: n_frames * n_src
            G[:] -= v2[:, :, None] @ G[:, None, s, :]


        demix_t(Y, X_tilde, P)
        demix_tmp(Y_tmp, X_tilde, G)
        np.power(abs(Y.transpose([1, 0, 2])), 2.0, out=Y2)

        for s in range(n_src):
            lambda_aux[s] = 1 / np.sqrt(np.mean(Y2[s, :, :]))
            W[:, :, s] *= lambda_aux[s]
            Y2[s, :, :] *= lambda_aux[s] ** 2
            R[s, :, :] *= lambda_aux[s] ** 2
            T[s, :, :] *= lambda_aux[s] ** 2


    Y = Y.transpose([2, 0, 1]).copy()
    z = projection_back(Y, X_original[:, :, 0])
    Y *= np.conj(z[None, :, :])




Z = Y.transpose(1, 2, 0)

# baseline
from nara_wpe.wpe import wpe
Z_wpe = wpe(X,    taps=taps,    delay=delay,    iterations=5,    statistics_mode='full')
#######################################################################################################################
z = istft(Z.transpose(1, 2, 0), size=stft_options['size'], shift=stft_options['shift'])
for i in range(n_sources):
    sf.write('test_iss_'+str(i)+'.wav', z[i] , 16000)