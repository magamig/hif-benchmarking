import numpy as np
import scipy
from functions import blur_downsample,roundNum


#### this part mainly for the twice-optimization of TONWMD

def twice_optimization_with_estBR(X, Y, Z, B, R, k=31, ratio=8, lamb=1e-6, mu=1e-6):
    '''
    pre-optimization and post-optimization of TONWMD
    this is suitable for bluring by fft
    :param X:
    :param Y:
    :param Z:
    :param B:
    :param R:
    :param k:
    :param ratio:
    :param lamb:
    :param mu:
    :return:
    '''
    h, w, c = X.shape
    X = np.reshape(X, [h * w, -1], order='F').T
    Z = np.reshape(Z, [h * w, -1], order='F').T

    Y = np.reshape(Y, [roundNum(h / ratio) * roundNum(w / ratio), -1], order='F').T
    _, _, V = scipy.linalg.svd(X.T, full_matrices=False)
    P = V.T[:, :k]
    RP = np.dot(R, P)
    H1 = np.dot(RP.T, RP) + lamb * np.dot(P.T, P)
    H2 = np.dot(RP.T, Z) + lamb * np.dot(P.T, X)
    A = np.linalg.solve(H1, H2)

    ABD = blur_downsample(A, ratio, B, h, w)

    H3 = np.dot(ABD, ABD.T) + mu * np.dot(A, A.T)
    H4 = np.dot(Y, ABD.T) + mu * np.dot(X, A.T)
    P = np.linalg.solve(H3.T, H4.T)
    P = P.T

    target = np.dot(P, A)
    target = np.reshape(target.T, [h, w, -1], order='F')
    target = np.float32(target)

    return target
