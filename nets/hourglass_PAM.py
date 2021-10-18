import numpy as np
import torch
import torch.nn as nn


class convolution(nn.Module):
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(convolution, self).__init__()
    pad = (k - 1) // 2
    self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
    self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    conv = self.conv(x)
    bn = self.bn(conv)
    relu = self.relu(bn)
    return relu

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False) # 首先降低维度
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False) # 维度进行还原

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        avg_out2 = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))

        out = avg_out + max_out + avg_out2 
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False) # concat完channel维度为3
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # 在dim=1的维度上进行mean操作。
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out2 = torch.mean(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out, avg_out2], dim=1) # 沿着channel维度concat一块
        x = self.conv1(x)
        return self.sigmoid(x)

class residual(nn.Module):
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(residual, self).__init__()

    self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
    self.bn1 = nn.BatchNorm2d(out_dim)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
    self.bn2 = nn.BatchNorm2d(out_dim)

    self.skip = nn.Sequential(nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
                              nn.BatchNorm2d(out_dim)) \
      if stride != 1 or inp_dim != out_dim else nn.Sequential()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    conv1 = self.conv1(x)
    bn1 = self.bn1(conv1)
    relu1 = self.relu1(bn1)

    conv2 = self.conv2(relu1)
    bn2 = self.bn2(conv2)

    skip = self.skip(x)
    return self.relu(bn2 + skip)


# inp_dim -> out_dim -> ... -> out_dim # 降维
def make_layer(kernel_size, inp_dim, out_dim, modules, layer, stride=1):
  layers = [layer(kernel_size, inp_dim, out_dim, stride=stride)]
  layers += [layer(kernel_size, out_dim, out_dim) for _ in range(modules - 1)]
  return nn.Sequential(*layers)


# inp_dim -> inp_dim -> ... -> inp_dim -> out_dim # 升维
def make_layer_revr(kernel_size, inp_dim, out_dim, modules, layer):
  layers = [layer(kernel_size, inp_dim, inp_dim) for _ in range(modules - 1)]
  layers.append(layer(kernel_size, inp_dim, out_dim))
  return nn.Sequential(*layers)


# key point layer
def make_kp_layer(cnv_dim, curr_dim, out_dim):
  return nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False),
                       nn.Conv2d(curr_dim, out_dim, (1, 1)))

class kp_module(nn.Module): 
  #kp_module指的是hourglass基本模块
  def __init__(self, n, dims, modules):
    super(kp_module, self).__init__()

    self.n = n

    curr_modules = modules[0]
    next_modules = modules[1]

    curr_dim = dims[0]
    next_dim = dims[1]

    # curr_mod x residual，curr_dim -> curr_dim -> ... -> curr_dim
    self.top = make_layer(3, curr_dim, curr_dim, curr_modules, layer=residual) 


    self.down = nn.Sequential() 
    # curr_mod x residual，curr_dim -> next_dim -> ... -> next_dim
    self.low1 = make_layer(3, curr_dim, next_dim, curr_modules, layer=residual, stride=2) #降维

    self.ca2 = ChannelAttention(next_dim) # 堆叠注意力模块
    self.sa2 = SpatialAttention()

    # next_mod x residual，next_dim -> next_dim -> ... -> next_dim
    if self.n > 1: # 通过递归完成构建
      self.low2 = kp_module(n - 1, dims[1:], modules[1:])
    else:
      self.low2 = make_layer(3, next_dim, next_dim, next_modules, layer=residual) 
    # curr_mod x residual，next_dim -> next_dim -> ... -> next_dim -> curr_dim
    self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_modules, layer=residual) # 升维

    self.up = nn.Upsample(scale_factor=2)
    
    self.ca = ChannelAttention(curr_dim) # 堆叠注意力模块
    self.sa = SpatialAttention()

  def forward(self, x):
    up1 = self.top(x)

    down = self.down(x)
    low1 = self.low1(down)

    out12 = self.ca2(low1) # 并联堆叠注意力模块
    out22 = self.sa2(low1)
    out12 = out12*low1 # element-wise multiplication
    out22 = out22*low1
    low1 = out12 + out22

    low2 = self.low2(low1)
    low3 = self.low3(low2)
    up2 = self.up(low3)

    out1 = self.ca(up2) # 并联堆叠注意力模块
    out2 = self.sa(up2) 
    out1 = out1*up2 # element-wise multiplication
    out2 = out2*up2
    up2 = out1 + out2

    return up1 + up2


class exkp(nn.Module):
  '''
  整体模型调用
  large hourglass stack为2
  small hourglass stack为1
  n这里控制的是hourglass的阶数，以上两个都用的是5阶的hourglass
  exkp(n=5, nstack=1, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4]),
  '''
  def __init__(self, n, nstack, dims, modules, cnv_dim=256, num_classes=3):
    super(exkp, self).__init__()

    self.nstack = nstack # 堆叠多次hourglass
    self.num_classes = num_classes

    curr_dim = dims[0]
    # 快速降维为原来的1/4
    self.pre = nn.Sequential(convolution(7, 3, 128, stride=2),
                             residual(3, 128, curr_dim, stride=2))
    # 堆叠nstack个hourglass
    self.kps = nn.ModuleList([kp_module(n, dims, modules) for _ in range(nstack)])

    self.cnvs = nn.ModuleList([convolution(3, curr_dim, cnv_dim) for _ in range(nstack)])

    self.inters = nn.ModuleList([residual(3, curr_dim, curr_dim) for _ in range(nstack - 1)])

    self.inters_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                                                nn.BatchNorm2d(curr_dim))
                                  for _ in range(nstack - 1)])
    self.cnvs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                                              nn.BatchNorm2d(curr_dim))
                                for _ in range(nstack - 1)])
    
    self.ca = ChannelAttention(cnv_dim) 
    self.sa = SpatialAttention()
    
    # heatmap layers heatmap输出通道为num_classes
    self.hmap = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])
    for hmap in self.hmap:
      hmap[-1].bias.data.fill_(-2.19) # -2.19是focal loss中的默认参数，-ln((1-pi)/pi),这里的pi取0.1

    # regression layers
    self.regs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)]) # 回归的输出通道为2
    self.w_h_ = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)]) # w和h
    self.pxpy = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)]) # w和h

    self.relu = nn.ReLU(inplace=True)

  def forward(self, image):
    inter = self.pre(image) 

    outs = []
    for ind in range(self.nstack): # 堆叠hourglass
      kp = self.kps[ind](inter)
      cnv = self.cnvs[ind](kp)

      x1 = self.ca(cnv) # 并联堆叠注意力模块
      x2 = self.sa(cnv) 
      x1 = x1*cnv # element-wise multiplication
      x2 = x2*cnv
      cnv = x1 + x2

      if self.training or ind == self.nstack - 1:
        outs.append([self.hmap[ind](cnv), self.regs[ind](cnv), self.w_h_[ind](cnv), self.pxpy[ind](cnv)])

      if ind < self.nstack - 1:
        inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
        inter = self.relu(inter)
        inter = self.inters[ind](inter)
    return outs


get_hourglass = \
  {'large_hourglass':
     exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4]),
   'small_hourglass':
     exkp(n=5, nstack=1, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4])}

if __name__ == '__main__':
  from collections import OrderedDict
  from utils.utils import count_parameters, count_flops, load_model


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    # pass


  net = get_hourglass['large_hourglass']
  load_model(net, '../ckpt/pretrain/checkpoint.t7')
  count_parameters(net)
  count_flops(net, input_size=512)

  for m in net.modules():
    if isinstance(m, nn.Conv2d):
      m.register_forward_hook(hook)

  with torch.no_grad():
    y = net(torch.randn(2, 3, 512, 512).cuda())
  # print(y.size())
