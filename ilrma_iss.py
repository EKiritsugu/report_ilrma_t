# Copyright (c) 2020 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Implementation of AuxIVA with independent source steering (ISS) updates
"""
import numpy as np



def project_back(Y, ref):
    """
    This function computes the frequency-domain filter that minimizes
    the squared error to a reference signal. This is commonly used
    to solve the scale ambiguity in BSS.

    .. note::

        Despite its name, this function is an implementation of the minimum
        distortion principle described in the two following papers.
        The bad naming comes from the confusion of the author of this code
        (myself) with a different method. m(_ _)m

        - K. Matsuoka and S. Nakashima, “Minimal distortion principle for blind
          source separation,” in Proc. ICA, San Diego, Dec. 2001, pp. 722–727.
        - K.Matsuoka,“Minimal distortion principle for blind source separation,” in
          Proc. SICE, Aug. 2002, pp. 2138–2143.

    Derivation of the projection
    ----------------------------

    The optimal filter `z` minimizes the squared error.

    .. math::

        \min E[|z^* y - x|^2]

    It should thus satsify the orthogonality condition
    and can be derived as follows

    .. math::

        0 & = E[y^*\\, (z^* y - x)]

        0 & = z^*\\, E[|y|^2] - E[y^* x]

        z^* & = \\frac{E[y^* x]}{E[|y|^2]}

        z & = \\frac{E[y x^*]}{E[|y|^2]}

    In practice, the expectations are replaced by the sample
    mean.

    Parameters
    ----------
    Y: array_like (n_frames, n_bins, n_channels)
        The STFT data to project back on the reference signal
    ref: array_like (n_frames, n_bins)
        The reference signal
    """

    num = np.sum(np.conj(ref[:, :, None]) * Y, axis=0)
    denom = np.sum(np.abs(Y) ** 2, axis=0)
    c = num / np.maximum(1e-15, denom)

    return np.conj(c[None, :, :]) * Y



def projection_back(Y, ref, clip_up=None, clip_down=None):
    """
    This function computes the frequency-domain filter that minimizes
    the squared error to a reference signal. This is commonly used
    to solve the scale ambiguity in BSS.

    Here is the derivation of the projection.
    The optimal filter `z` minimizes the squared error.

    .. math::

        \min E[|z^* y - x|^2]

    It should thus satsify the orthogonality condition
    and can be derived as follows

    .. math::

        0 & = E[y^*\\, (z^* y - x)]

        0 & = z^*\\, E[|y|^2] - E[y^* x]

        z^* & = \\frac{E[y^* x]}{E[|y|^2]}

        z & = \\frac{E[y x^*]}{E[|y|^2]}

    In practice, the expectations are replaced by the sample
    mean.

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


def ilrma_iss(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    W0=None,
    n_components=2,
    return_filters=False,
    callback=None,
):
    """
    Implementation of ILRMA algorithm without partitioning function for BSS presented in

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, *Determined blind
    source separation unifying independent vector analysis and nonnegative matrix
    factorization,* IEEE/ACM Trans. ASLP, vol. 24, no. 9, pp. 1626-1641, Sept. 2016

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, and H. Saruwatari *Determined
    Blind Source Separation with Independent Low-Rank Matrix Analysis,* in
    Audio Source Separation, S. Makino, Ed. Springer, 2018, pp. 125-156.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the observed signal n_src: int, optional
        The number of sources or independent components
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for demixing matrix
    n_components: int
        Number of components in the non-negative spectrum
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix W (nfrequencies, nchannels, nsources)
    if ``return_filters`` keyword is True.
    """
    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    # Only supports determined case
    assert n_chan == n_src, "There should be as many microphones as sources"

    # initialize the demixing matrices
    # The demixing matrix has the following dimensions (nfrequencies, nchannels, nsources),
    if W0 is None:
        W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)
    else:
        W = W0.copy()

    # initialize the nonnegative matrixes with random values
    T = np.array(0.1 + 0.9 * np.random.rand(n_src, n_freq, n_components))
    V = np.array(0.1 + 0.9 * np.random.rand(n_src, n_frames, n_components))
    R = np.zeros((n_src, n_freq, n_frames))
    I = np.eye(n_src, n_src)
    U = np.zeros((n_freq, n_src, n_chan, n_chan), dtype=X.dtype)
    product = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    lambda_aux = np.zeros(n_src)
    eps = 1e-15
    eyes = np.tile(np.eye(n_chan, n_chan), (n_freq, 1, 1))

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_src, n_frames), dtype=X.dtype)
    X_original = X
    X = X.transpose([1, 2, 0]).copy()

    np.matmul(T, V.swapaxes(1, 2), out=R)

    v_iss = np.zeros((n_freq, n_src), dtype=X.dtype)
    n_frames_sqrt = np.sqrt(n_frames)

    # Compute the demixed output
    def demix(Y, X, W):
        Y[:, :, :] = np.matmul(W, X)

    demix(Y, X, W)

    # P.shape == R.shape == (n_src, n_freq, n_frames)
    P = np.power(abs(Y.transpose([1, 0, 2])), 2.0)
    iR = 1 / R

    for epoch in range(n_iter):
        # simple loop as a start
        for s in range(n_src):
            ## NMF
            ######

            T[s, :, :] *= np.sqrt(
                np.dot(P[s, :, :] * iR[s, :, :] ** 2, V[s, :, :])
                / np.dot(iR[s, :, :], V[s, :, :])
            )
            T[T < eps] = eps

            R[s, :, :] = np.dot(T[s, :, :], V[s, :, :].T)
            R[R < eps] = eps
            iR[s, :, :] = 1 / R[s, :, :]

            V[s, :, :] *= np.sqrt(
                np.dot(P[s, :, :].T * iR[s, :, :].T ** 2, T[s, :, :])
                / np.dot(iR[s, :, :].T, T[s, :, :])
            )
            V[V < eps] = eps

            R[s, :, :] = np.dot(T[s, :, :], V[s, :, :].T)
            R[R < eps] = eps
            iR[s, :, :] = 1 / R[s, :, :]

            ## IVA 说实话这段IVA看着有一丢丢怪，用到了文中不需要的参量，反正我重写一个吧
            ######
        for s in range(n_src):#修改一下，这一行去掉于不去掉区别重大
            v_num = (Y * iR.swapaxes(0,1)) @ np.conj(Y[:, s, :, None])  # (n_freq, n_src, 1)
            # OP: n_frames * n_src
            v_denom = iR.swapaxes(0,1) @ np.abs(Y[:, s, :, None]) ** 2
            # (n_freq, n_src, 1)

            # OP: n_src
            v = v_num[:, :, 0] / v_denom[:, :, 0]
            # OP: 1
            v[:, s] = 1.0 - np.sqrt(n_frames) / np.sqrt(v_denom[:, s, 0])

            # update demixed signals
            # OP: n_frames * n_src
            W[:] -= v[:, :, None] * W[:, None, s, :]

            ## 这下面是失败的ISS
            # v_num = (Y * iR.swapaxes(0,1)) @ np.conj(
            #     Y[:, s, :, None]
            # )  # (n_freq, n_src, 1)
            # v_denom = iR.swapaxes(0,1) @ np.abs(Y[:, s, :, None]) ** 2
            # # (n_freq, n_src, 1)
            # v_iss[:, :] = v_num[:, :, 0] / v_denom[:, :, 0]
            # v_iss[:, s] =  1 - n_frames_sqrt / np.sqrt(v_denom[:, s, 0])
            # # update demixed signals
            # Y[:, :, :] -= v_iss[:, :, None] * Y[:, s, None, :]

            # Compute Auxiliary Variable
            # shape: (n_freq, n_chan, n_chan)

            # 这下面是IP
            # C = np.matmul((X * iR[s, :, None, :]), np.conj(X.swapaxes(1, 2))) / n_frames
            #
            # WV = np.matmul(W, C)
            # W[:, s, :] = np.conj(np.linalg.solve(WV, eyes[:, :, s]))
            #
            # # normalize
            # denom = np.matmul(
            #     np.matmul(W[:, None, s, :], C[:, :, :]), np.conj(W[:, s, :, None])
            # )
            # W[:, s, :] /= np.sqrt(denom[:, :, 0])

        demix(Y, X, W)

            ###一切修改发生在这个之间

        np.power(abs(Y.transpose([1, 0, 2])), 2.0, out=P)
        P[P < eps] = eps

        for s in range(n_src):
            lambda_aux[s] = 1 / np.sqrt(np.mean(P[s, :, :]))

            Y[:, s, :] *= lambda_aux[s]
            # W[:, :, s] *= lambda_aux[s]
            P[s, :, :] *= lambda_aux[s] ** 2
            R[s, :, :] *= lambda_aux[s] ** 2
            T[s, :, :] *= lambda_aux[s] ** 2

    Y = Y.transpose([2, 0, 1]).copy()

    if proj_back:
        z = projection_back(Y, X_original[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, W
    else:
        return Y





def auxiva_iss(
    X,
    n_iter=20,
    proj_back=True,
    model="laplace",
    return_filters=False,
    callback=None,
    callback_checkpoints=[],
    **kwargs,
):


    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    n_src = X.shape[2]

    # for now, only supports determined case
    assert n_chan == n_src

    # pre-allocate arrays
    r_inv = np.zeros((n_src, n_frames))
    v_num = np.zeros((n_freq, n_src), dtype=X.dtype)
    v_denom = np.zeros((n_freq, n_src), dtype=np.float64)
    v_iss = np.zeros((n_freq, n_src), dtype=X.dtype)

    # Things are more efficient when the frequencies are over the first axis
    X = X.transpose([1, 2, 0]).copy()

    # Initialize the demixed outputs
    Y = X.copy()

    n_frames_sqrt = np.sqrt(n_frames)

    for epoch in range(n_iter):

        # shape: (n_src, n_frames)
        # OP: n_frames * n_src
        eps = 1e-10
        if model == "laplace":
            r_inv[:, :] = 1.0 / np.maximum(eps, 2.0 * np.linalg.norm(Y, axis=0))
        elif model == "gauss":
            r_inv[:, :] = 1.0 / np.maximum(
                eps, (np.linalg.norm(Y, axis=0) ** 2) / n_freq
            )

        # Update now the demixing matrix
        for s in range(n_src):

            # OP: n_frames * n_src
            v_num = (Y * r_inv[None, :, :]) @ np.conj(
                Y[:, s, :, None]
            )  # (n_freq, n_src, 1)
            # OP: n_frames * n_src
            v_denom = r_inv[None, :, :] @ np.abs(Y[:, s, :, None]) ** 2
            # (n_freq, n_src, 1)

            # OP: n_src
            v_iss[:, :] = v_num[:, :, 0] / v_denom[:, :, 0]
            # OP: 1
            v_iss[:, s] -= n_frames_sqrt / np.sqrt(v_denom[:, s, 0])

            # update demixed signals
            # OP: n_frames * n_src
            Y[:, :, :] -= v_iss[:, :, None] * Y[:, s, None, :]

        # Monitor the algorithm progression
        if callback is not None and (epoch + 1) in callback_checkpoints:
            Y_tmp = Y.transpose([2, 0, 1]).copy()
            callback(Y_tmp, model)

    if return_filters is not None:
        # Demixing matrices were not computed explicitely so far,
        # do it here, if necessary
        W = Y[:, :, :n_chan] @ np.linalg.inv(X[:, :, :n_chan])

    Y = Y.transpose([2, 0, 1]).copy()
    X = X.transpose([2, 0, 1])

    if proj_back:
        Y = project_back(Y, X[:, :, 0])

    if return_filters:
        return Y, W
    else:
        return Y
