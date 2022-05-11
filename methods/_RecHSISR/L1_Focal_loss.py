import torch
import warnings


class _Loss(torch.nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction
            

def l1_focal_loss(input, target, weight, reduction='mean'):
    if not (target.size() == input.size()):
        warnings.warn(
            "Input size({}) is not same with target size({}).".format(target.size(), input.size()),
            stacklevel=2
        )

    ret = torch.abs(input - target) * weight
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)

    return ret


class L1_Focal_Loss(_Loss):
    def __init__(self, reduction='mean'):
        super(L1_Focal_Loss, self).__init__(reduction)

    def forward(self, input, target, weight):
        return l1_focal_loss(input, target, weight)
