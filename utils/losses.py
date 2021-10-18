import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _neg_loss_slow(preds, targets):
  pos_inds = targets == 1  
  neg_inds = targets < 1  

  neg_weights = torch.pow(1 - targets[neg_inds], 4)

  loss = 0
  for pred in preds:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(preds, targets):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      preds (B x c x h x w)
      gt_regr (B x c x h x w)
  '''
  pos_inds = targets.eq(1).float()
  neg_inds = targets.lt(1).float()

  neg_weights = torch.pow(1 - targets, 4)

  loss = 0
  for pred in preds:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss / len(preds)


def _reg_loss(regs, gt_regs, mask):
  mask = mask[:, :, None].expand_as(gt_regs).float() 
  loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
  return loss / len(regs)

def _SmoothL1Loss(regs, gt_regs, mask):

  mask = mask[:, :, None].expand_as(gt_regs).float()
  criteria = nn.SmoothL1Loss(reduction='sum')
  loss = sum(criteria(r * mask, gt_regs * mask) / (mask.sum() + 1e-4) for r in regs)

  return loss / len(regs)

def _NewLoss(regs, gt_regs, mask):

  mask = mask[:, :, None].expand_as(gt_regs).float()

  for r in regs:
    print(gt_regs)
    x1 = torch.mul(r[:,:,0],torch.cos(r[:,:,1]))
    x2 = torch.mul(gt_regs[:,:,0],torch.cos(gt_regs[:,:,1]))
    y1 = torch.mul(r[:,:,0],torch.sin(r[:,:,1]))
    y2 = torch.mul(gt_regs[:,:,0],torch.sin(gt_regs[:,:,1]))
    mask = mask[:,:,0]
    los = torch.sqrt((x2-x1)*(x2-x1)* mask + (y2-y1)*(y2-y1)* mask)
    loss = los.sum() / (mask.sum() + 1e-4)
  return loss 