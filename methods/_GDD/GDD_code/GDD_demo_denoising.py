# =============================================================================
# import lib
# =============================================================================
from __future__ import print_function
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.image as mpimg
import torch
import torch.optim
from models.GDD_denoising import gdd
from utils.sr_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

# =============================================================================
# load data
# =============================================================================
path_to_data = 'data/flash_noflash/'
n_im = 2
guide_np = mpimg.imread(path_to_data + 'im_flash.jpg')
input_np = mpimg.imread(path_to_data + 'im_noflash.jpg')
guide_np = guide_np.astype(np.float32) / 255
input_np = input_np.astype(np.float32) / 255

# =============================================================================
# show input and guidance images
# =============================================================================
figsize = 10
fig = plt.figure(figsize=(figsize,figsize))
plt.imshow(input_np)
fig = plt.figure(figsize=(figsize,figsize))
plt.imshow(guide_np)

# =============================================================================
# Set parameters and net
# =============================================================================
input_depth = input_np.shape[2]

method =    '2D'
pad   =     'reflection'
OPT_OVER =  'net'

show_every = 1000 #500 
save_every = 1000

num_c = 32
LR = 0.01#try 0.01 0.001 0.0001

OPTIMIZER = 'adam'

num_iter = 1001#try 12000, 8000
reg_noise_std = 0.01  # try 0 0.03 0.05 0.08
mse_history = np.zeros(num_iter)
thresh_v = 0.01#0.000005, 0.00001
n_layer = 5
# layer size for each depth
im_layer_size = []
w,h = guide_np.shape[0], guide_np.shape[1]
for i in range(n_layer):
    im_layer_size.append([w,h])
    w, h = math.ceil(w/2), math.ceil(h/2)
net_param_fin = []

# =============================================================================
# Set net
# =============================================================================

# set input
net_input = get_noise(input_depth, method, ( math.ceil(guide_np.shape[0]/(2**n_layer)), math.ceil(guide_np.shape[1]/(2**n_layer))) ).type(dtype).detach()
# number of channels
input_depth = net_input.shape[1]

# define network structure  
net = gdd(input_depth, input_np.shape[2],
           num_channels_down = num_c,
           num_channels_up =   num_c,
           num_channels_skip =    num_c,  
           filter_size_up = 3, filter_size_down = 3, filter_skip_size=1,
           upsample_mode='bilinear', # downsample_mode='avg',
           need1x1_up=False,
           need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU', im_layer_size = im_layer_size).type(dtype)

# define MSE loss
mse = torch.nn.MSELoss().type(dtype)

# convert numpy to torch
input_torch = torch.from_numpy(input_np.transpose(2,0,1)).type(dtype)
input_torch = input_torch[None, :].cuda()
guide_np_t = guide_np.transpose(2,0,1)
msi_torch = torch.from_numpy(guide_np_t[None, :]).type(dtype)

# =============================================================================
# Define closure and optimize
# =============================================================================
mse_last = 0#1000
last_net = [None] * num_iter
mse_history = [None] * num_iter
back_p = 0
repeat = 0
def closure(ind_iter):
    global i, net_input, mse_last, last_net, back_p, repeat
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        
    
    out = net(msi_torch, net_input)
    total_loss = mse(out, input_torch)
       
    mse_i = total_loss.data.cpu().numpy()
    total_loss.backward()
        
    # Log
    if i % 100 == 0:
        print ('Iteration %05d   MSE_gap %.7f' % (i, (mse_i - mse_last)))
    
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
            mse_history[i] = mse_i
            i += 1
            
            
            
    if i % show_every == 0:
        out = out.detach().cpu().squeeze().numpy().transpose(1,2,0)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))
        ax1.imshow(guide_np, cmap='gray')
        ax2.imshow(input_np)
        ax3.imshow(out)
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

# save a final output and network parameters for each image
out = net(msi_torch, net_input)
out_np = out.detach().cpu().squeeze().numpy().transpose(1,2,0)
net_input_np = net_input.detach().cpu().squeeze().numpy().transpose(1,2,0)
net_param_fin.append(list(net.parameters()))
    
np.save("result/flash_noflash/out", out_np)
np.save("result/flash_noflash/rand_input.npy", net_input_np)
np.savez("result/flash_noflash/param.npz", np.array(net_param_fin))

