
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy

from utils.utils import _gather_feature, _tranpose_and_gather_feature, flip_tensor


def _nms(heat, kernel=3):
  hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
  keep = (hmax == heat).float()
  return heat * keep


def _topk(scores, K=40):
  batch, cat, height, width = scores.size() # batch，C, H ,w

  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys = (topk_inds / width).int().float()
  topk_xs = (topk_inds % width).int().float()

  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_ind / K).int()
  topk_inds = _gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

# 该函数的作用是将heat_map解码成b-box
def ctdet_decode(hmap, regs, w_h_, pxpy, K=100):
  batch, cat, height, width = hmap.shape # C,W和H
  # height , width = 128
  hmap=torch.sigmoid(hmap)


  # 这里，test的batch是 1
  # if flip test
  if batch > 1:
    hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2 
    w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2 # w_h_ 第一列是宽度，第二列是高度。
    regs = regs[0:1] 

  batch = 1

  hmap = _nms(hmap)  # perform nms on heatmaps

  scores, inds, clses, ys, xs = _topk(hmap, K=K)

  regs = _tranpose_and_gather_feature(regs, inds)
  regs = regs.view(batch, K, 2)
  xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
  ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

  w_h_ = _tranpose_and_gather_feature(w_h_, inds) 
  w_h_ = w_h_.view(batch, K, 2) 

  pxpy = _tranpose_and_gather_feature(pxpy, inds) 
  pxpy = pxpy.view(batch, K, 2) 


  clses = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)

  width1 = torch.abs(torch.mul(pxpy[..., 0:1],torch.cos(pxpy[..., 1:2]))) # 半宽度
  height1 = torch.abs(torch.mul(pxpy[..., 0:1],torch.sin(pxpy[..., 1:2]))) # 半高度

  width1 = 0.1 * width1 + 0.9 * w_h_[..., 0:1]/2
  height1 = 0.1 * height1 + 0.9 * w_h_[..., 1:2] / 2


  bboxes = torch.cat([xs - width1,
                      ys - height1,
                      xs + width1,
                      ys + height1], dim=2)

  detections = torch.cat([bboxes, scores, clses], dim=2)
  return detections


