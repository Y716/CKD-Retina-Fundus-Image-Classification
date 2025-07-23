from .losses import FocalLoss, BCELoss, CELoss, DiceLoss, CE_DiceLoss

loss = {
  'focal-loss': FocalLoss,
  'bce-loss': BCELoss,
  'ce-loss': CELoss,
  'dice-loss': DiceLoss,
  'ce-dice-loss': CE_DiceLoss,
}