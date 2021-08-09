import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torchvision.models as models
import numpy as np
import sys
import torch.nn.functional as F

class GradCam():
    def __init__(self, model, percent=0.15, device='cpu'):
        self.model = model.to(device)
        self.bottleneck_out = None
        self.gradients = None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.device = device
        self.percent = percent
        
    def save_BottleneckGrad(self, grad):
        self.gradients = grad
        
    def forward_pass(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        bottleneck_num = 1
        for named, module in self.model.named_modules():
            if isinstance(module, models.resnet.Bottleneck):
                x = module(x)
                if bottleneck_num == 16:
                    self.bottleneck_out = x
                    x.register_hook(self.save_BottleneckGrad)
                bottleneck_num += 1
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x
    
    def get_outputs(self, inputs, label):
        self.model.eval()
        inputs = inputs.to(self.device)
        label = label.to(self.device)
        
        outputs = self.forward_pass(inputs)
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
        
        if have_gradient is None or self.gradients is None or self.bottleneck_out is None:
            return None
        
        weights = self.avgpool(self.gradients)
        sm = weights * self.bottleneck_out
            
#         sm = torch.sum(self.relu(sm.squeeze(dim=0)), dim=0).cpu().detach().numpy()
#         sm = (sm - np.min(sm)) / (np.max(sm) - np.min(sm))
#         sm = np.uint8(sm * 255)
#         sm = np.uint8(Image.fromarray(sm).resize((224, 224), Image.ANTIALIAS)) / 255

        sm = torch.sum(self.relu(sm.squeeze(dim=0)), dim=0, keepdim=True).unsqueeze(0)
        sm = F.interpolate(sm, size=(224, 224), mode='bilinear', align_corners=False).squeeze(dim=0).squeeze(0)
        sm = sm.cpu().detach().numpy()
        sm = (sm - np.min(sm)) / (np.max(sm) - np.min(sm))
        
        return sm
    
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
        
        
        