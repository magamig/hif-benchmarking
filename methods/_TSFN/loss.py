import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.L1_Loss = nn.L1Loss()

    def forward(self, output, label):
        l1_loss = self.L1_Loss(output, label)
        return l1_loss
        
class Spectral_Loss(nn.Module):
    def __init__(self):
        super(Spectral_Loss, self).__init__()

    def forward(self, output, rgb, srf):
        rgb_re = torch.matmul(output.permute(0, 2, 3, 1), srf.permute(1, 0)).permute(0, 3, 1, 2)
        spectral_loss = torch.pow((rgb_re - rgb), 2).mean()
        return spectral_loss

class Spatial_Loss(nn.Module):
    def __init__(self):
        super(Spatial_Loss, self).__init__()

    def forward(self, output, hsi_down, kernel):
        down_output = F.conv2d(output, kernel, groups=31, stride=8, padding=0)
        spatial_loss = torch.pow((down_output - hsi_down), 2).mean()
        return spatial_loss

class SAM_Loss(nn.Module):
	def __init__(self):
		super(SAM_Loss, self).__init__()

	def forward(self, output, label):
		ratio = (torch.sum((output+1e-8).mul(label+1e-8), dim=1)) / (torch.sqrt(torch.sum((output+1e-8).mul(output+1e-8), dim=1)*torch.sum((label+1e-8).mul(label+1e-8), dim=1)))
		angle = torch.acos(ratio)
		return torch.mean(angle)

