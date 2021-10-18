# TEST

import os
import sys
import cv2
import argparse
import numpy as np
from tqdm import tqdm
#import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.utils.data

#from datasets.coco import COCO_eval
#from datasets.pascal import PascalVOC_eval
from datasets.Damage import COCO_MEAN, COCO_STD, COCO_NAMES
from datasets.Damage import Damage, Damage_eval # your own data set

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
cfg.pretrain_dir = os.path.join(cfg.ckpt_dir, 'checkpoint.t7') # model

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
  
  dataset = Dataset_eval(cfg.data_dir, split='test', test_scales=cfg.test_scales, test_flip=cfg.test_flip) # split 为 test
  
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                            num_workers=1, pin_memory=True,
                                            collate_fn=dataset.collate_fn)
                                            
  print('Creating model...')
  if 'hourglass' in cfg.arch:
    model = get_hourglass[cfg.arch]
  elif 'resdcn' in cfg.arch:
    model = get_pose_net_resdcn(num_layers=18, head_conv=64, num_classes=3)
  elif cfg.arch == 'resnet':
    model = get_pose_net(num_layers=18, head_conv=64, num_classes=3) # 增加最基础的 ResNet18
  elif cfg.arch == 'res_CBAM':
    model = get_pose_net_resnet_CBAM(num_layers=18, head_conv=64, num_classes=3)
  elif cfg.arch == 'resnet_BCBAM2':
    model = get_pose_net_resnet_BCBAM2(num_layers=18, head_conv=64, num_classes=3)
  elif cfg.arch == 'resnet_SE':
    model = get_pose_net_resnet_SE(num_layers=18, head_conv=64, num_classes=3)


  model = load_model(model, cfg.pretrain_dir)
  model = model.to(cfg.device)
  model.eval()

  results = {}
  with torch.no_grad():
    for inputs in tqdm(data_loader):
      img_id, inputs,img_path = inputs[0]
      print('id%s ',img_id)
      
      detections = []
      for scale in inputs:
        inputs[scale]['image'] = inputs[scale]['image'].to(cfg.device)

        output = model(inputs[scale]['image'])[-1]
        dets = ctdet_decode(*output, K=cfg.test_topk) # bounding box
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]
        #print(len(dets)) 
        #print(dets.shape) 
        top_preds = {}
        dets[:, :2] = transform_preds(dets[:, 0:2],  
                                      inputs[scale]['center'],
                                      inputs[scale]['scale'],
                                      (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
        dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                       inputs[scale]['center'],
                                       inputs[scale]['scale'],
                                       (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
        cls = dets[:, -1]
        for j in range(dataset.num_classes):
          inds = (cls == j)
          top_preds[j + 1] = dets[inds, :5].astype(np.float32) 
          top_preds[j + 1][:, :4] /= scale
        
        detections.append(top_preds)
        #print(np.array(detections).shape)

      bbox_and_scores = {}
      for j in range(1, dataset.num_classes + 1):
        bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
        if len(dataset.test_scales) > 1:
          soft_nms(bbox_and_scores[j], Nt=0.5, method=2)
      scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, dataset.num_classes + 1)])

      if len(scores) > max_per_image: 
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, dataset.num_classes + 1):
          keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
          bbox_and_scores[j] = bbox_and_scores[j][keep_inds]
      
      images_test = cv2.imread(img_path)
      fig = plt.figure(0)
      colors = COCO_COLORS
      names = COCO_NAMES
      
      plt.imshow(cv2.cvtColor(images_test, cv2.COLOR_BGR2RGB))
      for lab in bbox_and_scores: 
        for boxes in bbox_and_scores[lab]: # five columns
          x1, y1, x2, y2, score = boxes
          if (x1<0):
             x1 = 0
          if (y1<0):
             y1 = 0
          if (y2>511):
            y2 = 510
          
          if score > 0.3:
            plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=colors[lab], facecolor='none'))
            plt.text(x1 -12 , y1 - 12 , names[lab], bbox=dict(facecolor=colors[lab], alpha=0.5), fontsize=7, color='k') # change the style

      
      fig.patch.set_visible(False)
      Save_dir = 'data/damage/Predict_images' # save images
      Image_name = img_path[-10:]
      Save_dir = os.path.join(Save_dir, Image_name)
      plt.axis('off')
      plt.savefig(Save_dir, dpi=400, transparent=True, bbox_inches="tight", pad_inches=0.1)
      plt.show()

      results[img_id] = bbox_and_scores 

  eval_results = dataset.run_eval(results, cfg.ckpt_dir)
  print(eval_results)


if __name__ == '__main__':
  main()
