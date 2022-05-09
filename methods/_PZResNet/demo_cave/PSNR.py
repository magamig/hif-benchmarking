import torch
def PSNR(hsi,hsi_g):
    mse = (hsi-hsi_g)**2
    mse = mse.float()
    mse = mse.mean(-1).mean(-1).mean(-1)
    psnr = torch.log10(mse**(-1)*(2**16)**2)*10
    return psnr.detach().cpu().item()