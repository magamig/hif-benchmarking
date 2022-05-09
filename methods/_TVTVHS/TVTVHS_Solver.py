###########TV-TV Solver for RGB Fusion Hyerspectral Image Super-Resolution##########
# Solves
#                              S_0
#                  minimize    Σ  TV(x_s) + beta*TV(x_s - w_s)
#                     X        s=1
#
#                   subject to Ax_s = z_s
#                              XR = Y
#
#  where TV(x_s) is the 2D total variation of a vectorized version of X_s
#  with dimensions M_0.N_0, z_s: m x 1 is a vector of measurements
#  (z_s = AX_s), Y: matrix with dimensions M_0.N_0 x S, beta > 0, and w_s is a
#  vectorized image similar to the image we want to reconstruct. We use ADMM to
#  solve the above problem, as explained in the paper. Access to A is given
#  implicitly through the function handler for the operations A*x and A'*y. In
#  this code, A is the averaging operation over non-overlapping blocks with
#  size (scaling_factor x scaling_factor).
#
# Inputs:
#    - M: number of rows in each band of the original image
#    - N: number of columns in each band of the original image (n = M*N)
#    - Z_s: matrix where each column s represents the sth vectorized (column major)
#           spectral band of Z_s (z_s = vec(Z_s))
#    - W_s: matrix where each column s represents the sth vectorized (column major)
#          spectral band of W_s (w_s = vec(W_s))
#    - beta: positive number
#    - Y: HRMS image with dimensions M_0.N_0 x S
#    - scaling_factor: scaling factor
#    - R: CSR function with dimensions S_0 x S
#
# Outputs:
#    - x_opt: solution of the above optimization problem
#
#  This code was designed and implemented by M. Vella to perform experiments
#  described in
#  [1] M. Vella, B. Zhang, W. Chen and J. F. C. Mota
#      Enhanced Hyperspectral Image Super-Resolution via RGB Fusion and TV-TV Minimization
#      preprint:
#      2021

# ========================================================================================
# TVTV_Solver: minimizing TV+TV with linear constraints
# Copyright (C) 2021  Marija Vella
# ========================================================================================
# Any feedback, comments, bugs and questions are welcome and can be sent to mv37@hw.ac.uk
#  =======================================================================================

# ========================================================================================

from numpy import linalg as LA
from utils import*
import multiprocessing
from joblib import delayed, Parallel
from numpy.linalg import inv


def get_u(nbands, lam, rho, w, u_bar, Fc_v, Fc_h, beta, v_bar):

    num_cores = multiprocessing.cpu_count()
    u = Parallel(n_jobs=num_cores)(delayed(find_u)(lam[:, i], rho[i], w[:,i], u_bar[:,i], Fc_v, Fc_h, beta[i], v_bar[:,i]) for i in range(nbands))

    return u


def get_v(nbands, n, lam, rho, x_bar, Fc_v, Fc_h, mu, h, u_bar, v_bar):

    num_cores = multiprocessing.cpu_count()
    v = Parallel(n_jobs=num_cores)(delayed(find_v)(n, lam[:, i], rho[i], x_bar[:,i], Fc_v, Fc_h,  mu[:, i], h, u_bar[:, i], v_bar[:, i]) for i in range(nbands))

    return v


def find_u(lam, rho, w, u_bar, Fc_v, Fc_h, beta, v_bar):

        u2 = u_bar.copy()
        w2 = w
        rhow = rho * w2
        rho2 = rho
        w_pos = (w2 >= 0)
        w_pos = w_pos * 1
        s = lam - (rho * D(v_bar, Fc_v, Fc_h))

        # Components for which w_i >= 0
        case1 = (w_pos == 1) & (s < -rhow - beta - 1)
        case1 = case1 * 1
        u2[np.nonzero(case1)[0]] = (-beta - 1 - s[np.nonzero(case1)[0]]) / rho2

        case2 = (w_pos == 1) & (-rhow - beta - 1 <= s) & (s <= -rhow + beta - 1)
        case2 = case2 * 1
        u2[np.nonzero(case2)[0]] = w2[(np.nonzero(case2))[0]]

        case3 = (w_pos == 1) & (-rhow + beta - 1 < s) & (s < beta - 1)
        case3 = case3 * 1
        u2[np.nonzero(case3)[0]] = (beta - 1 - s[np.nonzero(case3)[0]]) / rho2

        case4 = (w_pos == 1) & (beta - 1 <= s) & (s <= beta + 1)
        case4 = case4 * 1
        u2[np.nonzero(case4)[0]] = 0

        case5 = (w_pos == 1) & (s > beta + 1)
        case5 = case5 * 1
        u2[np.nonzero(case5)[0]] = (beta + 1 - s[np.nonzero(case5)[0]]) / rho2

        # Components for which w_i < 0
        case1r = (w_pos == 0) & (s < -beta - 1)
        case1r = case1r * 1
        u2[np.nonzero(case1r)[0]] = (-beta - 1 - s[np.nonzero(case1r)[0]]) / rho2

        case2r = (w_pos == 0) & (-beta - 1 <= s) & (s <= -beta + 1)
        case2r = case2r * 1
        u2[np.nonzero(case2r)[0]] = 0

        case3r = (w_pos == 0) & (-beta + 1 < s) & (s < -rhow - beta + 1)
        case3r = case3r * 1
        u2[np.nonzero(case3r)[0]] = (-beta + 1 - s[np.nonzero(case3r)[0]]) / rho2

        case4r = (w_pos == 0) & (-rhow - beta + 1 <= s) & (s <= -rhow + beta + 1)
        case4r = case4r * 1
        u2[np.nonzero(case4r)[0]] = w2[(np.nonzero(case4r))[0]]

        case5r = (w_pos == 0) & (s > -rhow + beta + 1)
        case5r = case5r * 1
        u2[np.nonzero(case5r)[0]] = (beta + 1 - s[np.nonzero(case5r)[0]]) / rho2
        u_bar = u2

        return u_bar


def find_v(n, lam, rho, x_bar, Fc_v, Fc_h, mu, h, u_bar, v_bar):

    v_bar_prev = v_bar
    g = DT(u_bar + ((1 / rho) * lam), Fc_v, Fc_h, n) + ((1 / rho) * mu) + x_bar
    v_aux = np.fft.ifft(h * np.fft.fft(g, axis=0), axis=0)
    v_bar = np.real(v_aux)

    return np.concatenate((np.real(v_bar), v_bar_prev))


def multiply_B(x, A, AT, scaling_factor, M, N, nbands):

    y1 = AT(x, scaling_factor, int(M/scaling_factor), int(N/scaling_factor), nbands)
    y = A(y1, scaling_factor, M, N, nbands)

    return y


def conjgrad(A2, b, x2, A, AT, scaling_factor, M, N, nbands):

    out = np.zeros((x2.shape[0], nbands, 1))
    MAX_ITER = 10000
    TOL = 10e-5
    r_full = b - np.squeeze(A2(x2, A, AT, scaling_factor, M, N, nbands))
    x2_full = x2

    for k in range(nbands):
        x2 = np.expand_dims(x2_full[:, k], 1)
        r = np.expand_dims(r_full[:, k], 1)
        f = r
        rsold = np.dot(np.transpose(r), r)

        for i in range(MAX_ITER):
            Ap = A2(f, A, AT, scaling_factor, M, N, 1)
            alpha = rsold / np.dot(np.transpose(f), Ap)
            x2 = x2 + alpha * f
            r = r - alpha*Ap
            rsnew = np.dot(np.transpose(r), r)

            if np.sqrt(rsnew) < TOL:
                out[:, k] = x2
                break

            f = r + rsnew / rsold * f
            rsold = rsnew

    return out


def TVTVHS_Solver(M, N, z, w_im, beta, scaling_factor, R, y):

    nbands = z.shape[1]  # no. of bands in HS image
    n = M * N
    m = len(z)
    beta = np.repeat(beta, nbands)

    MAX_ITER = 120  # maximum no. of ADMM iterations
    rho = np.full((nbands, 1), 0.2)

    A = lambda x, scaling_factor, M, N, nbands: A_h_multi(x, scaling_factor, M, N, nbands)
    AT = lambda x, scaling_factor, M, N, nbands: AT_h_multi(x, scaling_factor, M, N, nbands)

    # Vectors c_h and c_v defining the circulant matrices
    c_h = np.zeros((n, 1))
    c_h[0] = -1
    c_h[n - M] = 1

    c_v = np.zeros((n, 1))
    c_v[0] = -1
    c_v[n-1] = 1

    Fc_h = np.fft.fft(c_h, axis=0)
    Fc_v = np.fft.fft(c_v,  axis=0)

    # Squaring the diagonal matrices Fc_h and Fc_v
    Fc_v_diag = np.real(Fc_v)
    Fc_v_diag_square = Fc_v_diag ** 2  # vector containing diagonal entries squared

    Fc_h_diag = np.real(Fc_h)
    Fc_h_diag_square = Fc_h_diag ** 2  # vector containing diagonal entries squared
    h = 1 / (Fc_v_diag_square + Fc_h_diag_square + 1)
    #  -----------------------------------------------------------------------------------------------------------

    #  -----------------------------------------------------------------------------------------------------------

    B = lambda z, A, AT, scaling_factor, M, N, nbands: multiply_B(z, A, AT, scaling_factor, M, N, nbands)

    # Transform the side information w_im into the domain of derivatives
    w = np.zeros((2*n, nbands, 1))

    for i in range(nbands):
        w[:, i] = D(w_im[:, i], Fc_v, Fc_h)

    x_bar = w_im
    u_bar = w
    v_bar = x_bar

    lam = np.zeros(((2*n), nbands, 1))
    mu = np.zeros((n, nbands, 1))
    r_prim = np.zeros(((3*n), nbands, 1))
    s_dual = np.zeros(((3 * n), nbands, 1))
    gamma_p9 = np.zeros((m, nbands))

    for k in range(0, MAX_ITER):

        # Minimization in u, each band solved in parallel
        u_bar = get_u(nbands, lam, rho, w, u_bar, Fc_v, Fc_h, beta, v_bar)
        u_bar = np.swapaxes(np.asarray(u_bar), 0, 1)

        # Minimization in X, using matrices (individual bands can't be processed in parallel)
        p = (1 / rho) * ((rho * v_bar) - mu)
        p = np.squeeze(p)
        z = np.squeeze(z)
        
        # Lagrange multipliers gamma and eta
        gamma_p2 = z - np.squeeze(A(p, scaling_factor, M, N, nbands))
        gamma_p3 = np.squeeze(A(p, scaling_factor, M, N, nbands))@R
        gamma_p4 = np.squeeze(A(y, scaling_factor, M, N, 3))
        gamma_p5 = gamma_p3 - gamma_p4
        gamma_p6 = inv(np.transpose(R)@R)@np.transpose(R)
        gamma_p7 = gamma_p5@gamma_p6
        gamma_p8 = gamma_p2 - gamma_p7
        gamma_p9 = -np.squeeze(conjgrad(B, gamma_p8, gamma_p9, A, AT, scaling_factor, M, N, nbands))
        gamma_p10 = inv(R@inv(np.transpose(R)@R)@np.transpose(R) + np.eye(nbands))
        gamma = np.squeeze(gamma_p9)@gamma_p10

        eta_p1 = AT(gamma, scaling_factor, M//scaling_factor, N//scaling_factor, nbands)
        eta_p2 = p@R - np.squeeze(eta_p1)@R - y
        eta_p3 = inv(np.transpose(R)@R)
        eta = -eta_p2@eta_p3

        x_1 = AT(gamma, scaling_factor, M//scaling_factor, N//scaling_factor, nbands)
        x = p - np.squeeze(x_1) + eta@np.transpose(R)
        x_bar = x

        x_bar = np.expand_dims(x_bar, 2)

        # Minimization in v, each band solved in parallel
        v_bar_prev = v_bar
        v_all = get_v(nbands, n, lam, rho, x_bar, Fc_v, Fc_h, mu, h, u_bar, v_bar)
        v_bar = np.swapaxes(np.asarray(v_all)[:,0:n],0,1)

        for j in range(nbands):
            
            #  Update dual variable
            r_prim[0:(2*n), j] = u_bar[:, j] - D(v_bar[:, j], Fc_v, Fc_h)  # primal residual
            r_prim[2*n:3*n, j] = x_bar[:, j] - v_bar[:, j]
            lam[:, j] = lam[:, j] + rho[j] * r_prim[0: 2*n, j]
            mu[:, j] = mu[:, j] + rho[j] * r_prim[2*n:3*n, j]

            s_dual[0:2*n, j] = -rho[j] * (D(v_bar[:, j] + v_bar_prev[:, j], Fc_v, Fc_h))  # dual residual
            s_dual[2*n:3*n, j] = -rho[j] * (v_bar[:, j] + v_bar_prev[:, j])

            #  rho adjustment
            r_prim_norm = LA.norm(r_prim[:, j], ord=2)
            s_dual_norm = LA.norm(s_dual[:, j], ord=2)

            if k >= MAX_ITER:
                print('Warning: Maximum number of iterations reached. Primal residual = %f, Dual residual = %f\n', r_prim_norm, s_dual_norm)

    x_opt = np.real(x_bar)

    return x_opt
