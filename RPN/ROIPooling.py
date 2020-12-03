#https://github.com/escorciav/roi_pooling  for Faster R-CNN
#https://blog.csdn.net/shanglianlm/article/details/102002844

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

x = torch.randn(1, 1, 10, 10) * 10
print(x)
y1 = F.adaptive_max_pool2d(x, (5, 5))
print(y1)

roi_pool = ops.RoIPool(output_size=(5, 5), spatial_scale=1)
rois = torch.tensor([[0, 0, 0, 9, 9]], dtype=torch.float)
y2 = roi_pool(x, rois)
print(y2)

rois = [torch.tensor([[0, 0, 9, 9]], dtype=torch.float)]
y3 = roi_pool(x, rois)
print(y3)
"""
import torch
from torchvision.ops import RoIAlign

if __name__ == '__main__':
    output_size = (3,3)
    spatial_scale = 1/4 
    sampling_ratio = 2  

    #x.shape:(1,1,6,6)
    x = torch.FloatTensor([[
        [[1,2,3,4,5,6],
        [7,8,9,10,11,12],
        [13,14,15,16,17,18],
        [19,20,21,22,23,24],
        [25,26,27,28,29,30],
        [31,32,33,34,35,36],],
    ]])

    rois = torch.tensor([
        [0,0.0,0.0,20.0,20.0],
    ])
    channel_num = x.shape[1]
    roi_num = rois.shape[0]

    a = RoIAlign(output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
    ya = a(x, rois)
    print(ya)
"""