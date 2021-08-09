import os
from PIL import Image, ImageDraw
import sys
import zipfile
import xml.etree.ElementTree as ET
import argparse
import numpy as np
from PIL import ImageFilter
import matplotlib.cm as mpl_color_map
import copy
import torch.nn as nn
import torch
from scipy.ndimage import gaussian_laplace,gaussian_filter

from tqdm import tqdm

def compute_iou(bbox1, bbox2):
    x_min = torch.max(bbox1[0], bbox2[0])
    x_max = torch.min(bbox1[2], bbox2[2])
    y_min = torch.max(bbox1[1], bbox2[1])
    y_max = torch.min(bbox1[3], bbox2[3])
    area_cross = (x_max - x_min) * (y_max - y_min)
    area_bbox1 = ((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    if (x_max - x_min < 0) or (y_max - y_min < 0):
        return 0
    else:
        return area_cross / (area_bbox1 + area_bbox2 - area_cross)
    
    
def compute_EBPG(sm, bbox1, bbox2):
    mask = torch.zeros(sm.size())
    x_min = int(torch.max(bbox1[0], bbox2[0]))
    x_max = int(torch.min(bbox1[2], bbox2[2]))
    y_min = int(torch.max(bbox1[1], bbox2[1]))
    y_max = int(torch.min(bbox1[3], bbox2[3]))
    if (x_max - x_min < 0) or (y_max - y_min < 0):
        return 0
    mask[y_min:y_max, x_min:x_max] = 1
    tmp = (mask * sm).sum()
    return tmp / sm.sum()
    
# def compute_EBPG(sm, bbox1, bbox2, idx=0):
#     mask = torch.zeros(sm.size())
#     x_min = int(torch.max(bbox1[0], bbox2[0]))
#     x_max = int(torch.min(bbox1[2], bbox2[2]))
#     y_min = int(torch.max(bbox1[1], bbox2[1]))
#     y_max = int(torch.min(bbox1[3], bbox2[3]))
    
#     mask[y_min:y_max, x_min:x_max] = 1
#     tmp = (mask * sm).sum()
    
#     color_map = mpl_color_map.get_cmap('jet')
#     no_trans_heatmap = color_map(sm)
#     heatmap = copy.copy(no_trans_heatmap)
#     heatmap[:, :, 3] = 0.4
#     heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
#     if heatmap.mode in ('RGBA', 'P'):
#         heatmap = heatmap.convert('RGB')
        
    
#     draw = ImageDraw.Draw(heatmap)
#     draw.line([(bbox1[0], bbox1[1]),
#                (bbox1[0], bbox1[3]),
#                (bbox1[2], bbox1[3]),
#                (bbox1[2], bbox1[1]),
#                (bbox1[0], bbox1[1])], fill=(255, 0, 0), width=5)
#     draw.line([(bbox2[0], bbox2[1]),
#                (bbox2[0], bbox2[3]),
#                (bbox2[2], bbox2[3]),
#                (bbox2[2], bbox2[1]),
#                (bbox2[0], bbox2[1])], fill=(120, 0, 0), width=5)
#     draw.line([(x_min, y_min),
#                (x_min, y_max),
#                (x_max, y_max),
#                (x_max, y_min),
#                (x_min, y_min)], fill=(255, 255, 255), width=5)
#     heatmap.save('sample/'  + str(idx) + '__.jpg' )
    
#     save_headmp((mask*sm).numpy(), 'jet', idx=idx)
    
    
    
#     return tmp / sm.sum()

class LoGLayer(nn.Module):
    def __init__(self, in_channel, sigma):
        super(LoGLayer, self).__init__()
        self.sigma=sigma
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(2), 
            nn.Conv2d(in_channel, in_channel, 5, stride=1, padding=0, bias=None, groups=in_channel)
        )
        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((5,5))
        n[2,2] = 1
        k = gaussian_laplace(n,sigma=self.sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))

def save_headmp(activation, colormap_name, bbox=None, b_box=None, bbb_box=None, idx=0):
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    if heatmap.mode in ('RGBA', 'P'):
        heatmap = heatmap.convert('RGB')
        
    
#     draw = ImageDraw.Draw(heatmap)
#     draw.line([(bbox[0], bbox[1]),
#                (bbox[0], bbox[3]),
#                (bbox[2], bbox[3]),
#                (bbox[2], bbox[1]),
#                (bbox[0], bbox[1])], fill=(255, 0, 0), width=5)
#     draw.line([(b_box[0], b_box[1]),
#                (b_box[0], b_box[3]),
#                (b_box[2], b_box[3]),
#                (b_box[2], b_box[1]),
#                (b_box[0], b_box[1])], fill=(120, 0, 0), width=5)
#     draw.line([(bbb_box[0], bbb_box[1]),
#                (bbb_box[0], bbb_box[3]),
#                (bbb_box[2], bbb_box[3]),
#                (bbb_box[2], bbb_box[1]),
#                (bbb_box[0], bbb_box[1])], fill=(255,255,255), width=5)
    
    
    heatmap.save('sample/'  + str(idx) + '.jpg' )
    
    
def save_headmp2(activation, colormap_name, idx=0):
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    if heatmap.mode in ('RGBA', 'P'):
        heatmap = heatmap.convert('RGB')
    heatmap.save('sample/'  + str(idx) + '.jpg' )

def get_bbox(xml_path):
    xmltree = ET.parse(xml_path) 
    filename = xmltree.find('filename').text
    wnid = filename.split('_')[0]
    image_id = filename.split('_')[1]
    objects = xmltree.findall('object')
    bound_box = None
    for object_iter in objects:
        name = object_iter.find('name').text
        if name == wnid:
            bndbox = object_iter.find('bndbox')
            bndbox = [int(it.text) for it in bndbox]
    return bndbox, wnid, image_id

def get_mutil_obj(xml_path):
    xmltree = ET.parse(xml_path) 
    filename = xmltree.find('filename').text
    wnid = filename.split('_')[0]
    image_id = filename.split('_')[1]
    objects = xmltree.findall('object')
    return len(objects)

# def denormalaze()
    
    