# =============================================================================
# import lib
# =============================================================================
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.optim
import scipy.io
from skimage.measure import compare_psnr
from models.bicubic import *
from models.GDD_pansharpening import gdd
from utils.sr_utils import *

def tv_loss2(img):
    """
    Compute total variation loss.
    
    """
    b, chan, height, width = img.size()
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = (h_variance + w_variance)/ height/width/chan
    return loss    

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


# =============================================================================
# load data
# =============================================================================
n_im = 4
path_to_data = 'data/PAN/'
GT = scipy.io.loadmat(path_to_data + 'pan_im' + str(n_im) + '.mat')['I_MS_GT']
pan_np = scipy.io.loadmat(path_to_data + 'pan_im' + str(n_im) + '.mat')['I_PAN_input']
msi_np = scipy.io.loadmat(path_to_data + 'pan_im' + str(n_im) + '.mat')['I_MS_input']
init_pan_torch = torch.from_numpy(pan_np[None, None, :]).type(dtype)

# =============================================================================
# show MSI and PAN images
# =============================================================================
figsize = 10
fig = plt.figure(figsize=(figsize,figsize))
plt.imshow(msi_np[:,:,[4,2,1]])
fig = plt.figure(figsize=(figsize,figsize))
plt.imshow(pan_np)

# =============================================================================
# Set parameters and net
# =============================================================================
input_depth = msi_np.shape[2]
param_balance_name = '01'
param_balance = 0.1
net_param_fin = []
factor = 32
method =    '2D'
pad   =     'reflection'
OPT_OVER =  'net'

show_every = 1000 #500 
save_every = 1000

num_c = 64
LR = 0.01#try 0.01 0.001 0.0001

OPTIMIZER = 'adam'

num_iter = 8001#try 6000, 8000
reg_noise_std = 0.01  # try 0 0.03 0.05 0.08
mse_history = np.zeros(num_iter)
n_layer = 5
thresh_v = 0.1#
param_balance = 10#10 for PAN and 100 for full resolution
param_balance_name = '10'
# layer size for each depth
im_layer_size = []
w,h = pan_np.shape[0], pan_np.shape[1]
for i in range(n_layer):
    im_layer_size.append([w,h])
    w, h = math.ceil(w/2), math.ceil(h/2)

# =============================================================================
# Set net
# =============================================================================

# set input
net_input = get_noise(input_depth, method, ( math.ceil(pan_np.shape[0]/(2**n_layer)), math.ceil(pan_np.shape[1]/(2**n_layer))) ).type(dtype).detach()

# number of channels
input_depth = net_input.shape[1]

# define network structure  
net = gdd(input_depth, msi_np.shape[2],
           num_channels_down = num_c,
           num_channels_up =   num_c,
           num_channels_skip =    num_c,  
           filter_size_up = 3, filter_size_down = 3, filter_skip_size=1,
           upsample_mode='bilinear', 
           need1x1_up=False,
           need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU', im_layer_size = im_layer_size).type(dtype)

# define MSE loss
mse = torch.nn.MSELoss().type(dtype)

# convert numpy to torch
pan_torch = torch.from_numpy(pan_np).type(dtype)
pan_torch_expand = pan_torch.unsqueeze(0).repeat(input_depth,1,1)
pan_torch_expand = pan_torch_expand [None,:]
msi_torch = torch.from_numpy(msi_np.transpose(2,0,1)).type(dtype)
msi_torch = msi_torch[None, :].cuda()

# define downsampler
bicubic_func = bicubic()

# =============================================================================
# Define closure and optimize
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

    
    out_HR, out_PAN = net(init_pan_torch, net_input)
    out_LR = bicubic_func(out_HR, scale = (1/4))
    total_loss = mse(out_LR, msi_torch)*param_balance + tv_loss2(out_PAN - pan_torch_expand)
        
    
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
        out = out_HR.detach().cpu().squeeze().numpy().transpose(1,2,0)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))
        ax1.imshow(pan_np, cmap='gray')
        ax2.imshow(GT[:,:,[4,2,1]])
        ax3.imshow(out[:,:,[4,2,1]])
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
out, out_PAN = net(init_pan_torch, net_input)
out_HR_np = out.detach().cpu().squeeze().numpy().transpose(1,2,0)
net_input_np = net_input.detach().cpu().squeeze().numpy().transpose(1,2,0)
net_param_fin.append(list(net.parameters()))
    
scipy.io.savemat("result/PAN/out.mat",{'out': out_HR_np})
np.save("result/PAN/rand_input.npy", net_input_np)
np.savez("result/PAN/param.npz", np.array(net_param_fin))

