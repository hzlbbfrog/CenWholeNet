import os
import cv2
import json
import math
import numpy as np

import torch
import torch.utils.data as data
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

import sys
sys.path.append(r"E:\CenWholeNet\utils")
from image import get_border, get_affine_transform, affine_transform, color_aug
from image import draw_umich_gaussian, gaussian_radius

COCO_NAMES = ['__background__', 'Crack', 'Reinforcement Exposure', 'Spalling'] 
 
COCO_IDS = [1, 2, 3] 

COCO_MEAN = [0.34637799, 0.3756352, 0.36720032] # be not used
COCO_STD = [0.05302383, 0.05460293, 0.05676473] # 
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]


class Damage(data.Dataset): # Damage
  def __init__(self, data_dir, split, split_ratio=1.0, img_size=512): # img_size
    super(Damage, self).__init__()
    self.num_classes = 3 # modify
    self.class_name = COCO_NAMES
    self.valid_ids = COCO_IDS
    self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}  # cat_ids {1: 0, 2: 1, 3: 2}

    self.data_rng = np.random.RandomState(123)
    self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
    self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
    self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
    self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]

    self.split = split
    self.data_dir = os.path.join(data_dir, 'damage') # data set
    self.img_dir = os.path.join(self.data_dir, 'images') # image
    if split == 'test':
      self.annot_path = os.path.join(self.data_dir, 'annotations', 'test_damage.json')
    elif split == 'train':
      self.annot_path = os.path.join(self.data_dir, 'annotations', 'train_damage.json')
    else:
      self.annot_path = os.path.join(self.data_dir, 'annotations', 'val_damage.json')

    self.max_objs = 128 
    self.padding = 127  
    self.down_ratio = 4
    self.img_size = {'h': img_size, 'w': img_size}
    self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio}
    self.rand_scales = np.arange(0.6, 1.4, 0.1)
    self.gaussian_iou = 0.7

    print('==> initializing Damage %s data.' % split)
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds() 

    if 0 < split_ratio < 1:
      split_size = int(np.clip(split_ratio * len(self.images), 1, len(self.images)))
      self.images = self.images[:split_size]

    self.num_samples = len(self.images) # number

    print('Loaded %d %s samples' % (self.num_samples, split))

  def __getitem__(self, index): 
    img_id = self.images[index] # id
    img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name']) 
    ann_ids = self.coco.getAnnIds(imgIds=[img_id]) 
    annotations = self.coco.loadAnns(ids=ann_ids) # label
    labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations]) 
    bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32) 
    if len(bboxes) == 0: 。
      bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
      labels = np.array([[0]])
    bboxes[:, 2:] += bboxes[:, :2]  # x1 y1 w h to x1 y1 x2 y2

    img = cv2.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image 
    scale = max(height, width) * 1.0

    flipped = False 

    trans_img = get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']]) 
    img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h'])) 

    img = img.astype(np.float32) / 255. # [0,1]

    img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

    # Ground Truth heatmap 
    trans_fmap = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']]) 

    # vectors
    hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap，size（3,96,96）
    w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
    pxpy = np.zeros((self.max_objs, 2), dtype=np.float32) # length and theta
    regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression

    # index
    inds = np.zeros((self.max_objs,), dtype=np.int64)
    ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

    # detections = []
    for k, (bbox, label) in enumerate(zip(bboxes, labels)): 
      #if flipped:
      #  bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_fmap) 
      bbox[2:] = affine_transform(bbox[2:], trans_fmap)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1) 
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0] # h and w

      d = math.sqrt((bbox[3]-bbox[1])*(bbox[3]-bbox[1])+(bbox[2]-bbox[0])*(bbox[2]-bbox[0]))/2
      theta = math.pi-math.atan(h/w)

      if h > 0 and w > 0:
        obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32) 
        obj_c_int = obj_c.astype(np.int32) 

        radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou))) # gaussian_radius
        draw_umich_gaussian(hmap[label], obj_c_int, radius) 
        w_h_[k] = 1. * w, 1. * h
        pxpy[k] = 1. * d, 1. * theta 
        regs[k] = obj_c - obj_c_int  # discretization error 
        inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0] # = fmap_w * cy + cx 
        ind_masks[k] = 1 

    return {'image': img,
            'hmap': hmap, 'w_h_':w_h_,'pxpy': pxpy, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
            'c': center, 's': scale, 'img_id': img_id}

  def __len__(self):
    return self.num_samples


class Damage_eval(Damage):
  def __init__(self, data_dir, split, test_scales=(1,), test_flip=False, fix_size=False):
    super(Damage_eval, self).__init__(data_dir, split)
    self.test_flip = test_flip
    self.test_scales = test_scales
    self.fix_size = fix_size

  def __getitem__(self, index):
    img_id = self.images[index]
    img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
    image = cv2.imread(img_path)
    height, width = image.shape[0:2]

    out = {}
    for scale in self.test_scales:
      new_height = int(height * scale)
      new_width = int(width * scale)

      img_height, img_width = self.img_size['h'], self.img_size['w'] 
      center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      scaled_size = max(height, width) * 1.0
      scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)

      img = image
      img = img.astype(np.float32) / 255. # [0,1]
      img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

      out[scale] = {'image': img,
                    'center': center,
                    'scale': scaled_size,
                    'fmap_h': img_height // self.down_ratio,
                    'fmap_w': img_width // self.down_ratio}

    return img_id, out, img_path

  def convert_eval_format(self, all_bboxes): 
    # all_bboxes: num_samples x num_classes x 5
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]: 
        category_id = self.valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out = list(map(lambda x: float("{:.2f}".format(x)), bbox[0:4]))

          detection = {"image_id": int(image_id),
                       "category_id": int(category_id),
                       "bbox": bbox_out,
                       "score": float("{:.2f}".format(score))}
          detections.append(detection)
    return detections

  def run_eval(self, results, save_dir=None):
    detections = self.convert_eval_format(results) 

    if save_dir is not None:
      result_json = os.path.join(save_dir, "results.json")
      json.dump(detections, open(result_json, "w"))

    coco_dets = self.coco.loadRes(detections)
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

  @staticmethod
  def collate_fn(batch):
    out = []
    for img_id, sample, img_path in batch:
      out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
      if k == 'image' else np.array(sample[s][k]) for k in sample[s]} for s in sample},img_path))
    return out


if __name__ == '__main__':
  from tqdm import tqdm
  import pickle

  dataset = Damage('E:/CenWholeNet/data/damage', 'images') # your own data set.
  for d in dataset:
    b1 = d

  pass

