# =============================================================================
# import lib
# =============================================================================
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
import scipy.io
from skimage.measure import compare_psnr
from models.downsampler import Downsampler_ave_block
from models.GDD import gdd
from utils.sr_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

# =============================================================================
# load data
# =============================================================================
data = scipy.io.loadmat('data/HS/data.mat')
msi_np = data['MSI']
hsi_np = data['HSI']
GT = data['GT']
R = data['R']

R_torch = torch.from_numpy(R).type(dtype)# R_torch - 3 by number of channels
img_hr = torch.from_numpy(msi_np).type(dtype)
GT_rgb =np.tensordot(R, GT, axes=((1),(2))).transpose(1,2,0)
init_msi_np = msi_np.transpose(2,0,1)
init_msi_torch = torch.from_numpy(init_msi_np[None, :]).type(dtype)


# =============================================================================
# show MSI and HSI
# =============================================================================
figsize = 10
fig = plt.figure(figsize=(figsize,figsize))
plt.imshow(hsi_np[:,:,[25,10,5]])
fig = plt.figure(figsize=(figsize,figsize))
plt.imshow(msi_np)

# =============================================================================
# Set parameters and net
# =============================================================================
input_depth = hsi_np.shape[2]
factor = 32
param_balance_name = '01'
param_balance = 0.1
net_param_fin = []
method =    '2D'
pad   =     'reflection'
OPT_OVER =  'net'

show_every = 1000 #500 
save_every = 1000

num_c = 64
LR = 0.01#try 0.01 0.001 0.0001

OPTIMIZER = 'adam'

num_iter = 20001#try 15000, 10000
reg_noise_std = 0.03  # try 0 0.03 0.05 0.08
mse_history = np.zeros(num_iter)
thresh_v = 0.00001#0.000005, 0.00001
n_layer = 5

# =============================================================================
# Set net
# =============================================================================

# set input
net_input = get_noise(input_depth, method, ( int(msi_np.shape[0]/(2**n_layer)), int(msi_np.shape[1]/(2**n_layer))) ).type(dtype).detach()

# number of channels
input_depth = net_input.shape[1]

# define network structure  
net = gdd(input_depth, hsi_np.shape[2],
           num_channels_down = num_c,
           num_channels_up =   num_c,
           num_channels_skip =    num_c,  
           filter_size_up = 3, filter_size_down = 3, filter_skip_size=1,
           upsample_mode='bilinear', # downsample_mode='avg',
           need1x1_up=False,
           need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

# define MSE loss
mse = torch.nn.MSELoss().type(dtype)

# convert numpy to torch
msi_torch = torch.from_numpy(msi_np).type(dtype)
hsi = hsi_np.transpose(2,0,1)
hsi_torch = torch.from_numpy(hsi).type(dtype)
hsi_torch = hsi_torch[None, :].cuda()

# define downsampler
downsampler = Downsampler_ave_block(kernel_size=factor, factor=factor).type(dtype)

# =============================================================================
# Define closure
# =============================================================================
mse_last = 0#1000
last_net = [None] * num_iter
psnr_history = [None] * num_iter
mse_history = [None] * num_iter
back_p = 0
repeat = 0
def closure(ind_iter):
    global i, net_input, mse_last, last_net, back_p, repeat
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out_HR = net(init_msi_torch, net_input)
    out_LR = downsampler(out_HR)
    
    out_rgb_torch = torch.bmm(out_HR.squeeze(0).permute(1,2,0),R_torch.squeeze(0).transpose(1,0).expand(out_HR.size(2),*R_torch.squeeze(0).transpose(1,0).size()))
    total_loss = mse(out_LR, hsi_torch)*param_balance + mse(out_rgb_torch, msi_torch)
        
    mse_i = total_loss.data.cpu().numpy()
    total_loss.backward()
        
    psnr_HR = compare_psnr(GT.astype(np.float32), out_HR.detach().cpu().squeeze().numpy().transpose(1,2,0))
    
    # Log
    if i % 100 == 0:
        print ('Iteration %05d    PSNR_HR %.3f MSE_gap %.7f' % (i, psnr_HR, (mse_i - mse_last)))
    
    # Track the loss function
    if (mse_i - mse_last) > thresh_v and i > 1000: 
        print('increase in the loss at the pixel of %05d.' % (i))
        print('MSE_gap %.7f' % (mse_i - mse_last))
        if back_p == 0:
            back_p = i-10#1
            repeat = 150
            
        for new_param, net_param in zip(last_net[back_p], net.parameters()):
            net_param.data.copy_(new_param)
        return total_loss*0
    else:
        if back_p > 0:
            i = back_p
            back_p = 0
            
        elif repeat > 0:
            repeat -= 1
            return total_loss
            
        else:
            last_net[i] = [x.detach().cpu() for x in net.parameters()]
            last_net[i-51] = None
            #last_net[i-101] = None
            
            if i>40:
                mse_last = np.mean(mse_history[i-40:i-20])
            # History
            psnr_history[i] = psnr_HR
            mse_history[i] = mse_i
            i += 1
            
            
            
    if i % show_every == 0:
        out_rgb =np.tensordot(R, out_HR.detach().cpu().squeeze().numpy(), axes=((1),(0))).transpose(1,2,0)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))
        ax1.imshow(msi_np[:,:,[2,1,0]], cmap='gray')
        ax2.imshow(GT_rgb[:,:,[2,1,0]])
        ax3.imshow(out_rgb[:,:,[2,1,0]])
        plt.show()
        
    return total_loss


# =============================================================================
# Optimization
# =============================================================================
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
i = 0
p = get_params(OPT_OVER, net, net_input)
print('Starting optimization with ADAM')
optimizer = torch.optim.Adam(p, lr=LR)
for j in range(num_iter):
    optimizer.zero_grad()
    total_loss = closure(j)
    optimizer.step()

psnr_history = np.asarray(psnr_history, dtype=np.float32)

# save a final output (mat file) and network parameters for each image
out = net(init_msi_torch, net_input)
out_HR_np = out.detach().cpu().squeeze().numpy().transpose(1,2,0)
net_input_np = net_input.detach().cpu().squeeze().numpy().transpose(1,2,0)
net_param_fin.append(list(net.parameters()))
    
scipy.io.savemat("result/HS/out_HR_HSI.mat",{'out': out_HR_np})
np.save("result/HS/rand_input.npy", net_input_np)
np.savez("result/HS/param.npz", np.array(net_param_fin))

