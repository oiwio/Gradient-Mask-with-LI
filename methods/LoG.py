import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
import random
import torch.nn.functional as F
from utils.utils import LoGLayer, save_headmp
import numpy as np
import sys

class LoG():
    def __init__(self, model, percent=0.15, device='cpu'):
        self.model = model.to(device)
        self.bw_mask = []
        self.layer_hook()
        self.gb = LoGLayer(1, 11).to(device)
        self.relu = nn.ReLU(inplace=True)
        self.trans = transforms.ToTensor()
        self.device = device
        self.percent = percent
        
    def layer_hook(self):
        
        def hook_bw(module, grad_in, grad_out):
            grad = grad_in[0]
            grad = torch.norm(grad, dim=1, keepdim=True)
            log_mask = self.gb(grad)
            
            log_mask = log_mask.view(log_mask.shape[2], log_mask.shape[3])
            self.bw_mask.append(log_mask)
            
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(hook_bw)
                
    def resize_tensor(self, tensor):
        size = (224, 224)
        tensor = tensor.cpu().numpy()
        tensor = Image.fromarray(tensor).resize(size, Image.ANTIALIAS)
        return self.trans(tensor).squeeze(dim=0)
    
        
    def get_outputs(self, inputs, label):
        self.model.eval()
        inputs = inputs.to(self.device)
        label = label.to(self.device)
        
        outputs = self.model(inputs)
        pred = outputs.argmax(dim=1)
        
        if pred != label:
            return None
        else:
            return outputs
        
    def get_gradient(self, inputs, label):
        self.model.eval()
        
        outputs = self.get_outputs(inputs, label)
        
        if outputs is None:
            return None
        
        target = torch.FloatTensor(1, outputs.size()[-1]).zero_().to(self.device)
        target[0][torch.argmax(outputs, dim=1)] = 1
        outputs.backward(gradient=target)
        
        return outputs
        
    def get_sm(self, inputs, label):
        have_gradient = self.get_gradient(inputs, label)
        
        if have_gradient is None or len(self.bw_mask) == 0:
            return None
        
        masks = [self.resize_tensor(t) for _, t in enumerate(self.bw_mask)]
        masks = torch.stack(masks) * -1.
        
#         masks = torch.sum(self.relu(masks),dim=0).squeeze(dim=0).squeeze(dim=0).numpy()
#         masks = (masks - np.min(masks)) / (np.max(masks) - np.min(masks))
#         masks = np.uint8(masks * 255)
#         masks = np.uint8(Image.fromarray(masks).resize((224,224), Image.ANTIALIAS)) / 255

        masks = torch.sum(self.relu(masks),dim=0, keepdim=True).unsqueeze(0)
        masks = F.interpolate(masks, size=(224, 224), mode='bilinear', align_corners=False).squeeze(dim=0).squeeze(0)
        masks = masks.numpy()
        masks = (masks - np.min(masks)) / (np.max(masks) - np.min(masks))
        
        self.bw_mask.clear()
        return masks
    
        
    def get_bbox(self, inputs, label):
        sm = self.get_sm(inputs, label)
        if sm is None:
            return None, None
        sm = torch.from_numpy(sm)
        kth = torch.kthvalue(sm.view(-1), int(sm.numel() * (1 - self.percent)))[0]
        masks = torch.where(sm > kth, torch.ones(1), torch.zeros(1))
        nozeros_mask = torch.nonzero(masks)
        xy_max = torch.max(nozeros_mask, dim=0)
        xy_min = torch.min(nozeros_mask, dim=0)
        x_min = xy_min[0][1]
        y_min = xy_min[0][0]
        x_max = xy_max[0][1]
        y_max = xy_max[0][0]
        b_box = []
        b_box.append(x_min)
        b_box.append(y_min)
        b_box.append(x_max)
        b_box.append(y_max)
        b_box = torch.FloatTensor(b_box)
        return b_box, sm

        
        
        