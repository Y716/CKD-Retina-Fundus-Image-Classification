import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        :param gamma: focusing parameter for modulating factor (1 - p_t)
        :param weight: class weights (Tensor of shape [num_classes])
        :param reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, **kwargs):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        """
        Standard CrossEntropyLoss for multi-class classification.
        :param weight: class weights
        :param ignore_index: index to ignore
        :param reduction: 'mean' or 'sum'
        """
        super(CELoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight,
                                           ignore_index=ignore_index,
                                           reduction=reduction)

    def forward(self, inputs, targets, **kwargs):
        return self.loss_fn(inputs, targets)
