##这个代码用于测试ILRMA-T-IP算法
# 已经成功

# ToDo ：对本代码进行函数化封装


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

    """Generate a new array that chops the given array along the given axis
     into overlapping frames.

    Note: if end='pad' the return is maybe a copy

    Args:
        x: The array to segment
        length: The length of each frame
        shift: The number of array elements by which to step forward
               Negative values are also allowed.
        axis: The axis to operate on; if None, act on the flattened array
        end: What to do with the last frame, if the array is not evenly
                divisible into pieces. Options are:
                * 'cut'   Simply discard the extra values
                * None    No end treatment. Only works when fits perfectly.
                * 'pad'   Pad with a constant value
                * 'conv_pad' Special padding for convolution, assumes
                             shift == 1, see example below
        pad_mode: see numpy.pad
        pad_value: The value to use for end='pad'

    Examples:
        >>> # import cupy as np
        >>> segment_axis(np.arange(10), 4, 2)  # simple example
        array([[0, 1, 2, 3],
               [2, 3, 4, 5],
               [4, 5, 6, 7],
               [6, 7, 8, 9]])
        >>> segment_axis(np.arange(10), 4, -2)  # negative shift
        array([[6, 7, 8, 9],
               [4, 5, 6, 7],
               [2, 3, 4, 5],
               [0, 1, 2, 3]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 1, axis=0)
        array([[0, 1, 2, 3],
               [1, 2, 3, 4]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 2, axis=0, end='cut')
        array([[0, 1, 2, 3]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 2, axis=0, end='pad')
        array([[0, 1, 2, 3],
               [2, 3, 4, 0]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 1, axis=0, end='conv_pad')
        array([[0, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 2],
               [0, 1, 2, 3],
               [1, 2, 3, 4],
               [2, 3, 4, 0],
               [3, 4, 0, 0],
               [4, 0, 0, 0]])
        >>> segment_axis(np.arange(6).reshape(6), 4, 2, axis=0, end='pad')
        array([[0, 1, 2, 3],
               [2, 3, 4, 5]])
        >>> segment_axis(np.arange(10).reshape(2, 5), 4, 1, axis=-1)
        array([[[0, 1, 2, 3],
                [1, 2, 3, 4]],
        <BLANKLINE>
               [[5, 6, 7, 8],
                [6, 7, 8, 9]]])
        >>> segment_axis(np.arange(10).reshape(5, 2).T, 4, 1, axis=1)
        array([[[0, 2, 4, 6],
                [2, 4, 6, 8]],
        <BLANKLINE>
               [[1, 3, 5, 7],
                [3, 5, 7, 9]]])
        >>> segment_axis(np.asfortranarray(np.arange(10).reshape(2, 5)),
        ...                 4, 1, axis=1)
        array([[[0, 1, 2, 3],
                [1, 2, 3, 4]],
        <BLANKLINE>
               [[5, 6, 7, 8],
                [6, 7, 8, 9]]])
        >>> segment_axis(np.arange(8).reshape(2, 2, 2).transpose(1, 2, 0),
        ...                 2, 1, axis=0, end='cut')
        array([[[[0, 4],
                 [1, 5]],
        <BLANKLINE>
                [[2, 6],
                 [3, 7]]]])
        >>> a = np.arange(7).reshape(7)
        >>> b = segment_axis(a, 4, -2, axis=0, end='cut')
        >>> a += 1  # a and b point to the same memory
        >>> b
        array([[3, 4, 5, 6],
               [1, 2, 3, 4]])

        >>> segment_axis(np.arange(7), 8, 1, axis=0, end='pad').shape
        (1, 8)
        >>> segment_axis(np.arange(8), 8, 1, axis=0, end='pad').shape
        (1, 8)
        >>> segment_axis(np.arange(9), 8, 1, axis=0, end='pad').shape
        (2, 8)
        >>> segment_axis(np.arange(7), 8, 2, axis=0, end='cut').shape
        (0, 8)
        >>> segment_axis(np.arange(8), 8, 2, axis=0, end='cut').shape
        (1, 8)
        >>> segment_axis(np.arange(9), 8, 2, axis=0, end='cut').shape
        (1, 8)

        >>> x = np.arange(1, 10)
        >>> filter_ = np.array([1, 2, 3])
        >>> np.convolve(x, filter_)
        array([ 1,  4, 10, 16, 22, 28, 34, 40, 46, 42, 27])
        >>> x_ = segment_axis(x, len(filter_), 1, end='conv_pad')
        >>> x_
        array([[0, 0, 1],
               [0, 1, 2],
               [1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6],
               [5, 6, 7],
               [6, 7, 8],
               [7, 8, 9],
               [8, 9, 0],
               [9, 0, 0]])
        >>> x_ @ filter_[::-1]  # Equal to convolution
        array([ 1,  4, 10, 16, 22, 28, 34, 40, 46, 42, 27])

        >>> segment_axis(np.arange(19), 16, 4, axis=-1, end='pad')
        array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
               [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,  0]])

        >>> import torch
        >>> segment_axis(torch.tensor(np.arange(10)), 4, 2)  # simple example
        tensor([[0, 1, 2, 3],
                [2, 3, 4, 5],
                [4, 5, 6, 7],
                [6, 7, 8, 9]])
        >>> segment_axis(torch.tensor(np.arange(10) + 1j), 4, 2)  # simple example
        tensor([[0.+1.j, 1.+1.j, 2.+1.j, 3.+1.j],
                [2.+1.j, 3.+1.j, 4.+1.j, 5.+1.j],
                [4.+1.j, 5.+1.j, 6.+1.j, 7.+1.j],
                [6.+1.j, 7.+1.j, 8.+1.j, 9.+1.j]], dtype=torch.complex128)
    """
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
    """

    Note: The returned y_tilde consumes a similar amount of memory as Y, because
        of tricks with strides. Usually the memory consumprion is K times
        smaller than the memory consumprion of a contignous array,

    >>> T, D = 20, 2
    >>> Y = np.arange(start=1, stop=T * D + 1).reshape([T, D]).T
    >>> print(Y)
    [[ 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39]
     [ 2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40]]
    >>> taps, delay = 4, 2
    >>> Y_tilde = build_y_tilde(Y, taps, delay)
    >>> print(Y_tilde.shape, (taps*D, T))
    (8, 20) (8, 20)
    >>> print(Y_tilde)
    [[ 0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35]
     [ 0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36]
     [ 0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33]
     [ 0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34]
     [ 0  0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31]
     [ 0  0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32]
     [ 0  0  0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29]
     [ 0  0  0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30]]
    >>> Y_tilde = build_y_tilde(Y, taps, 0)
    >>> print(Y_tilde.shape, (taps*D, T), Y_tilde.strides)
    (8, 20) (8, 20) (-8, 16)
    >>> print('Pseudo size:', Y_tilde.nbytes)
    Pseudo size: 1280
    >>> print('Reak size:', Y_tilde.base.base.base.base.nbytes)
    Reak size: 368
    >>> print(Y_tilde)
    [[ 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39]
     [ 2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40]
     [ 0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37]
     [ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38]
     [ 0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35]
     [ 0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36]
     [ 0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33]
     [ 0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34]]

    The first columns are zero because of the delay.

    """
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
    """
    Use np.linalg.solve with fallback to np.linalg.lstsq.
    Equal to np.linalg.lstsq but faster.

    Note: limited currently by A.shape == B.shape

    This function try's np.linalg.solve with independent dimensions,
    when this is not working the function fall back to np.linalg.solve
    for each matrix. If one matrix does not work it fall back to
    np.linalg.lstsq.

    The reason for not using np.linalg.lstsq directly is the execution time.
    Examples:
    A and B have the shape (500, 6, 6), than a loop over lstsq takes
    108 ms and this function 28 ms for the case that one matrix is singular
    else 1 ms.

    >>> def normal(shape):
    ...     return np.random.normal(size=shape) + 1j * np.random.normal(size=shape)

    >>> A = normal((6, 6))
    >>> B = normal((6, 6))
    >>> C1 = np.linalg.solve(A, B)
    >>> C2, *_ = np.linalg.lstsq(A, B)
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C1, C2)
    >>> np.testing.assert_allclose(C1, C3)
    >>> np.testing.assert_allclose(C1, C4)

    >>> A = np.zeros((6, 6), dtype=np.complex128)
    >>> B = np.zeros((6, 6), dtype=np.complex128)
    >>> C1 = np.linalg.solve(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.LinAlgError: Singular matrix
    >>> C2, *_ = np.linalg.lstsq(A, B)
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C2, C3)
    >>> np.testing.assert_allclose(C2, C4)

    >>> A = normal((3, 6, 6))
    >>> B = normal((3, 6, 6))
    >>> C1 = np.linalg.solve(A, B)
    >>> C2, *_ = np.linalg.lstsq(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.LinAlgError: 3-dimensional array given. Array must be two-dimensional
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C1, C3)
    >>> np.testing.assert_allclose(C1, C4)


    >>> A[2, 3, :] = 0
    >>> C1 = np.linalg.solve(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.LinAlgError: Singular matrix
    >>> C2, *_ = np.linalg.lstsq(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.LinAlgError: 3-dimensional array given. Array must be two-dimensional
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C3, C4)


    """

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

n_sources = 3
mixed_sig_path = 'mixed/'+str(n_sources) + 'ch/'
save_path = 'wped/'+str(n_sources) + 'ch/'

file_list = os.listdir(mixed_sig_path)

stft_options = dict(size=1024, shift=1024//4)
sampling_rate = 16000
delay = 2
iterations = 100
taps = 5

wav_name = file_list[4]
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
    print(np.shape(X))# n_freq, n_ch, n_frames

    np.matmul(T, V.swapaxes(1, 2), out=R)
    # Compute the demixed output


    # Y2.shape == R.shape == (n_src, n_freq, n_frames)
    Y2 = np.power(abs(Y.transpose([1, 0, 2])), 2.0)
    iR = 1 / R

    ##WPE
    X_line = build_y_tilde(Y, taps, delay)
    X_tilde = np.array([np.vstack((X[i], X_line[i]))
                           for i in range(n_freq)])
    P = np.zeros((n_freq, n_src, X_tilde.shape[1]),dtype=X.dtype)
    W_p = P[:, :, :n_src]
    W_p[:,:,:] = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)
    A_p = np.zeros((n_freq, X_tilde.shape[1]), dtype=X.dtype)
    eyes_t = np.tile(np.eye( X_tilde.shape[1],  X_tilde.shape[1], dtype = X.dtype), (n_freq, 1, 1))
    # print(P[0,:,:])
    # print(np.shape(X_tilde))


    def demix(Y, X, W):
        Y[:, :, :] = np.matmul(W, X)
    demix(Y, X, W)

    def demix_t(Y_wpe,X_tilde,P):
        Y_wpe[:, :, :] = np.matmul(P, X_tilde)
    demix_t(Y_wpe, X_tilde, P)
    ###########

    for epoch in range(n_iter):
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
            ##########

            ## WPE
            V_wpe = np.matmul((X_tilde * iR[s, :, None, :]), np.conj(X_tilde.swapaxes(1, 2))) / n_frames

            A_p[:, :n_src] = np.linalg.solve(W_p, eyes[:, :, s])

            V_inv = _stable_solve(V_wpe, eyes_t)
            P[:, s, :] = np.conj(_stable_solve(V_wpe, A_p[:, :, None])).squeeze()
            denom_wpe = np.conj(A_p[:, None, :]) @ V_inv @ A_p[:, :, None]
            P[:, s, :] /= np.sqrt(denom_wpe[:, :, 0])




            """
            ## IVA
            ######

            # Compute Auxiliary Variable
            # shape: (n_freq, n_chan, n_chan)
            C = np.matmul((X * iR[s, :, None, :]), np.conj(X.swapaxes(1, 2))) / n_frames

            # WV = np.matmul(W, C)
            # W[:, s, :] = np.conj(np.linalg.solve(WV, eyes[:, :, s]))

            W_ = np.linalg.solve(W, eyes[:, :, s])
            W[:, s, :] = np.conj(np.linalg.solve(C, W_))

            # normalize
            denom = np.matmul(
                np.matmul(W[:, None, s, :], C[:, :, :]), np.conj(W[:, s, :, None])
            )
            # print(np.sqrt(denom[:, :, 0]))
            W[:, s, :] /= np.sqrt(denom[:, :, 0])

        # demix(Y, X, W)
        """
        demix_t(Y, X_tilde, P)
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
    sf.write('test_wpe_'+str(i)+'.wav', z[i] , 16000)