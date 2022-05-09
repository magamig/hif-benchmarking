import skimage.transform
import os
from skimage.measure import block_reduce
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from matplotlib import pyplot as plt
import shutil
import gdown


def get_W():
    # Download sample images from Google Drive
    if not os.path.exists('./data/W'):
        os.makedirs('./data/W')

    url = 'https://drive.google.com/u/0/uc?id=1bf6kWyAgf1JSTy25xwoR9qDLWG-UQtu7&export=download'
    output = 'balloons_ms'
    gdown.download(url, output, quiet=False)
    shutil.move('balloons_ms', './data/W/balloons_ms')

    url = 'https://drive.google.com/u/0/uc?id=1Jfr16RWW7BiukLiyzqOEqO8A9TCLujjG&export=download'
    output = 'clay_ms'
    gdown.download(url, output, quiet=False)
    shutil.move('clay_ms', './data/W/clay_ms')

    url = 'https://drive.google.com/u/0/uc?id=15vTwIRimGoUxaE9JBGhHCTvI6EglRoYq&export=download'
    output = 'flowers_ms'
    gdown.download(url, output, quiet=False)
    shutil.move('flowers_ms', './data/W/flowers_ms')


def AT_h(x, scaling_factor, M, N):

    dim1 = int(N*scaling_factor)
    dim2 = int(M*scaling_factor)
    x_up = x.reshape((int(M), int(N)), order='F')
    y = skimage.transform.resize(x_up, (dim2, dim1), order=0)
    y = y.reshape((-1, 1), order='F')

    return y


def A_h(x, scaling_factor, M, N):

    x_down = x.reshape((M, N), order='F')
    y = skimage.measure.block_reduce(x_down, (scaling_factor, scaling_factor), np.average)
    y = y.reshape((-1, 1), order='F')

    return y


def A_h_multi(x, scaling_factor, M, N, nbands):

    if nbands==1:
        x_down = x.reshape((M, N), order='F')
        y = skimage.measure.block_reduce(x_down, (scaling_factor, scaling_factor), np.average)
        out = y.reshape((-1, 1), order='F')
    else:
        x_down = np.zeros((M, N, nbands, 1))
        out = np.zeros((x.shape[0]//scaling_factor**2, nbands, 1))
        for i in range(nbands):
            x_down[:, :, i] = np.expand_dims(x[:,i].reshape((M, N), order='F'), 2)
            y = skimage.measure.block_reduce(np.squeeze(x_down[:,:,i]), (scaling_factor, scaling_factor), np.average)
            out[:, i] = y.reshape((-1, 1), order='F')

    return out


def AT_h_multi(x, scaling_factor, M, N, nbands):
    #  This needs to change according to what method is used to create the LR input
    dim1 = int(N*scaling_factor)
    dim2 = int(M*scaling_factor)

    if nbands == 1:
        x_up = x.reshape((M, N), order='F')
        y = skimage.transform.resize(x_up, (dim2, dim1), order=0)
        out = y.reshape((-1, 1), order='F')
    else:
        x_up = np.zeros((M, N, nbands, 1))
        out = np.zeros((x.shape[0] * scaling_factor ** 2, nbands, 1))
        for i in range(nbands):
            x_up[:, :, i] = np.expand_dims(x[:,i].reshape((M, N), order='F'), 2)
            y = skimage.transform.resize(np.squeeze(x_up[:,:,i]), (dim2, dim1), order=0)
            out[:, i] = y.reshape((-1, 1), order='F')
    return out


def D(z, Fc_v, Fc_h):

    D_v = lambda z: np.real(np.fft.ifft(np.multiply(Fc_v, np.fft.fft(z, axis=0)), axis=0))
    D_h = lambda z: np.real(np.fft.ifft(Fc_h * np.fft.fft(z, axis=0), axis=0))
    D = np.concatenate((D_v(z), D_h(z)), axis=0)

    return D


def DT(z, Fc_v, Fc_h, n):

    DT_v = lambda z: np.real(np.fft.fft(Fc_v * np.fft.ifft(z, axis=0), axis=0))
    DT_h = lambda z: np.real(np.fft.fft(Fc_h * np.fft.ifft(z, axis=0), axis=0))
    DT = DT_v(z[0:n]) + DT_h(z[n:2 * n])

    return DT


def compare_mpsnr(x_true, x_pred):

    x_true, x_pred = x_true.astype(np.float64), x_pred.astype(np.float64)
    channels = x_true.shape[2]
    total_psnr = [peak_signal_noise_ratio(x_true[:, :, k], x_pred[:, :, k], data_range=np.max(x_true[:,:,k]) - np.min(x_true[:,:,k]))
                  for k in range(channels)]

    return np.mean(total_psnr)


def compare_mssim(x_true, x_pred, multichannel=True):

    channels = x_true.shape[2]
    x_true, x_pred = x_true.astype(np.float64), x_pred.astype(np.float64)
    mssim = [structural_similarity(x_true[:, :, i], x_pred[:, :, i], multichannel=multichannel)
            for i in range(channels)]

    return np.mean(mssim)


def find_rmse(img_tar, img_hr):

    ref = img_tar * 255.0
    tar = img_hr * 255.0
    lr_flags = tar < 0
    tar[lr_flags] = 0
    hr_flags = tar > 255.0
    tar[hr_flags] = 255.0

    diff = ref - tar;
    size = ref.shape
    rmse = np.sqrt(np.sum(np.sum(np.power(diff, 2))) / (size[0] * size[1]*size[2]))

    return rmse


def compare_sam(x_true, x_pred):

    num = 0
    sum_sam = 0
    x_true, x_pred = x_true.astype(np.float64), x_pred.astype(np.float64)
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
                sum_sam += np.arccos(
                    np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred)))
                num += 1
    sam_deg = (sum_sam / num) * 180 / np.pi

    return sam_deg


def compare_ergas(x_true, x_pred, ratio):

    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    sum_ergas = 0
    for i in range(x_true.shape[0]):
        vec_x = x_true[i]
        vec_y = x_pred[i]
        err = vec_x - vec_y
        r_mse = np.mean(np.power(err, 2))
        tmp = r_mse / (np.mean(vec_x)**2)
        sum_ergas += tmp

    return (100 / ratio) * (np.sqrt(sum_ergas / x_true.shape[0]))


def img_2d_mat(x_true, x_pred):

    h, w, c = x_true.shape
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    x_mat = np.zeros((c, h * w), dtype=np.float32)
    y_mat = np.zeros((c, h * w), dtype=np.float32)
    for i in range(c):
        x_mat[i] = x_true[:, :, i].reshape((1, -1),order='F')
        y_mat[i] = x_pred[:, :, i].reshape((1, -1), order='F')

    return x_mat, y_mat


def evaluate_metrics(inX, x_solver, W, sf):

    av_psnr_tvtv = compare_mpsnr(inX, x_solver)
    av_ssim_tvtv = compare_mssim(inX, x_solver, multichannel=True)
    av_sam_tvtv = compare_sam(inX, x_solver)
    av_rmse_tvtv = find_rmse(inX, x_solver)
    av_ergas_tvtv = compare_ergas(inX, x_solver, sf)

    av_psnr_net = compare_mpsnr(inX, W)
    av_ssim_net = compare_mssim(inX, W, multichannel=True)
    av_sam_net = compare_sam(inX, W)
    av_rmse_net = find_rmse(inX, W)
    av_ergas_net = compare_ergas(inX, W, sf)

    return av_psnr_tvtv, av_ssim_tvtv, av_sam_tvtv, av_rmse_tvtv, av_ergas_tvtv, av_psnr_net, av_ssim_net, av_sam_net, av_rmse_net, av_ergas_net

def plot_results(inX, x_solver, W, band):

    # Plot outputs
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 15))
    ax1.imshow(inX[:, :, band], cmap='jet')
    ax1.set_title('GT')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(W[:, :, band], cmap='jet')
    ax2.set_title('Network')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.imshow(x_solver[:, :, band], cmap='jet')
    ax3.set_title('Ours')
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.show()
