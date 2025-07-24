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

    def forward(self, inputs, targets):
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

    def forward(self, inputs, targets):
        return self.loss_fn(inputs, targets)


class BCELoss(nn.Module):
    def __init__(self, reduction="mean", pos_weight=1.0):
        super(BCELoss, self).__init__()
        if isinstance(pos_weight, (int, float)):
            pos_weight = torch.tensor(pos_weight)
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=pos_weight)

    def forward(self, prediction, targets):
        return self.bce_loss(prediction, targets.float())


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        intersection = 2 * torch.sum(prediction * target) + self.smooth
        union = torch.sum(prediction) + torch.sum(target) + self.smooth
        loss = 1 - intersection / union
        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self, reduction="mean", D_weight=0.5):
        super(CE_DiceLoss, self).__init__()
        self.DiceLoss = DiceLoss()
        self.BCELoss = BCELoss(reduction=reduction)
        self.D_weight = D_weight

    def forward(self, prediction, targets):
        return self.D_weight * self.DiceLoss(prediction, targets) + (1 - self.D_weight) * self.BCELoss(prediction,
                                                                                                       targets)