
# In comparsion with test.py and test2.py, this file just tests all models, and images aren't saved.

import os
import sys
import time
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.utils.data

#from datasets.coco import COCO_eval
#from datasets.pascal import PascalVOC_eval
from datasets.Damage import COCO_MEAN, COCO_STD, COCO_NAMES
from datasets.Damage import Damage, Damage_eval 

from nets.hourglass_PAM import get_hourglass
#from nets.hourglass import get_hourglass
from nets.resdcn import get_pose_net_resdcn
from nets.resnet import get_pose_net
from nets.resnet_CBAM import get_pose_net_resnet_CBAM 
from nets.resnet_PAM import get_pose_net_resnet_PAM
from nets.resnet_SE import get_pose_net_resnet_SE 

from utils.utils import load_model
from utils.image import transform_preds
from utils.summary import create_logger
from utils.post_process import ctdet_decode

from nms.nms import soft_nms

COCO_COLORS = sns.color_palette('hls', len(COCO_NAMES))

# Training settings
parser = argparse.ArgumentParser(description='centernet')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='pascal_resdcn18_512')
parser.add_argument('--dataset', type=str, default='Damage', choices=['coco', 'pascal','Damage'])
parser.add_argument('--arch', type=str, default='resnet')
parser.add_argument('--img_size', type=int, default=512) 
parser.add_argument('--test_flip', action='store_true')
parser.add_argument('--test_scales', type=str, default='1')  # 0.5,0.75,1,1.25,1.5
parser.add_argument('--test_topk', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=1)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]

def main():
  logger = create_logger(save_dir=cfg.log_dir)
  print = logger.info
  print(cfg)

  cfg.device = torch.device('cuda')
  torch.backends.cudnn.benchmark = False

  max_per_image = 100
  
  Dataset_eval = Damage_eval 
  dataset = Dataset_eval(cfg.data_dir, split='train', test_scales=cfg.test_scales, test_flip=cfg.test_flip) # split test
  
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                            num_workers=1, pin_memory=True,
                                            collate_fn=dataset.collate_fn)
                                            
  print('Creating model...')
  if 'hourglass' in cfg.arch:
    model = get_hourglass[cfg.arch]
  elif 'resdcn' in cfg.arch:
    model = get_pose_net_resdcn(num_layers=18, head_conv=64, num_classes=3)
  elif cfg.arch == 'resnet':
    model = get_pose_net(num_layers=18, head_conv=64, num_classes=3) 
  elif cfg.arch == 'res_CBAM':
    model = get_pose_net_resnet_CBAM(num_layers=18, head_conv=64, num_classes=3)
  elif cfg.arch == 'resnet_PAM':
    model = get_pose_net_resnet_PAM(num_layers=18, head_conv=64, num_classes=3)
  elif cfg.arch == 'resnet_SE':
    model = get_pose_net_resnet_SE(num_layers=18, head_conv=64, num_classes=3)


  def Evaluate(epoch,model):
      print('\n Evaluate@Epoch: %d' % epoch)

      start_time=time.clock()
      print('Start time %s Seconds' %start_time)

      model.eval()
      torch.cuda.empty_cache()
      max_per_image = 100

      results = {}
      with torch.no_grad():
        for inputs in data_loader:
          img_id, inputs, img_path = inputs[0]

          detections = []
          for scale in inputs:
            inputs[scale]['image'] = inputs[scale]['image'].to(cfg.device) # (1,3)
            output = model(inputs[scale]['image'])[-1] # hmap, regs, pxpy
            dets = ctdet_decode(*output, K=cfg.test_topk) # torch.cat([bboxes, scores, clses], dim=2)
            dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

            top_preds = {}
            dets[:, :2] = transform_preds(dets[:, 0:2],
                                          inputs[scale]['center'],
                                          inputs[scale]['scale'],
                                          (inputs[scale]['fmap_w'], inputs[scale]['fmap_h'])) 
            dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                          inputs[scale]['center'],
                                          inputs[scale]['scale'],
                                          (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
            clses = dets[:, -1]
            for j in range(dataset.num_classes):
              inds = (clses == j)
              top_preds[j + 1] = dets[inds, :5].astype(np.float32)
              top_preds[j + 1][:, :4] /= scale

            detections.append(top_preds)

          bbox_and_scores = {j: np.concatenate([d[j] for d in detections], axis=0)
                            for j in range(1, dataset.num_classes + 1)}
          scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, dataset.num_classes + 1)])
          if len(scores) > max_per_image:
            kth = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, dataset.num_classes + 1):
              keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
              bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

          results[img_id] = bbox_and_scores
      
      end_time=time.clock()
      
      eval_results = dataset.run_eval(results, save_dir=cfg.ckpt_dir) 
      print(eval_results)

      print('End time %s Seconds' %end_time)
      Run_time = end_time - start_time
      FPS = 100/Run_time # replace 100 with the number of images
      print('FPS %s ' %FPS)

      #summary_writer.add_scalar('Evaluate_mAP/mAP', eval_results[0], epoch)
      return eval_results[0]

  num_epochs = 60 # replace 60 with the number of epoch
  Max_mAP = 0 

  for epoch in range(1, num_epochs + 1):
    cfg.pretrain_dir = os.path.join(cfg.ckpt_dir, 'checkpoint_epoch'+str(epoch)+'.t7') # the address
    model = load_model(model, cfg.pretrain_dir)
    model = model.to(cfg.device)
    
    mAP = Evaluate(epoch,model)
    if mAP > Max_mAP :
      Max_mAP = mAP
      print('Max_AP=%s' %Max_mAP) 

if __name__ == '__main__':
  main()
