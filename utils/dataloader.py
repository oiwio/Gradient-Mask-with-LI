import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import json
from PIL import Image, ImageDraw
import random
import xml.etree.ElementTree as ET
import numpy as np
import random

def get_bbox(xml_path):
    xmltree = ET.parse(xml_path) 
    filename = xmltree.find('filename').text
    wnid = filename.split('_')[0]
    image_id = filename.split('_')[1]
    objects = xmltree.findall('object')
    if len(objects) > 1:
        return None
    bound_box = None
    for object_iter in objects:
        name = object_iter.find('name').text
        if name == wnid:
            bndbox = object_iter.find('bndbox')
            bndbox = [int(it.text) for it in bndbox]
    return bndbox

def get_ImgLabelXml(root_path, xmls_path):
    samples = []
    classes = sorted(os.listdir(root_path))
    classes_index = {}
    for i in range(len(classes)):
        classes_index[classes[i]] = i
    for i in range(len(classes)):
        for root, _, fnames in os.walk(os.path.join(root_path, classes[i])):
            for fname in sorted(fnames):
                img_path = os.path.join(root, fname)
                img_label = classes_index[classes[i]]
                xml_name = fname[:-5] + '.xml'
                xml_path = os.path.join(xmls_path, classes[i], xml_name)
                item = img_path, xml_path, img_label
                samples.append(item)
    return samples


class Dataset_obj(torch.utils.data.Dataset):
    def __init__(self, root_path, xmls_path, transform):
        self.samples = get_ImgLabelXml(root_path, xmls_path)
        self.transform = transform
        
        self.num = 0
    def upnum(self,idx):
        self.num=idx
        
    def __getitem__(self, index):
        img_path, xml_path, img_label = self.samples[index]
        bbox = get_bbox(xml_path)
        img = Image.open(img_path)

        x, y = img.size
        img = img.resize((224,224))
        
        x_scale = 224 / x
        y_scale = 224 / y
        bbox[0] *= x_scale
        bbox[2] *= x_scale
        bbox[1] *= y_scale
        bbox[3] *= y_scale
        bbox = [int(bbox[i]) for i in range(4)]
        bbox[2] = 224 if bbox[2] > 224 else bbox[2]
        bbox[3] = 224 if bbox[3] > 224 else bbox[3]
        
        img = self.transform(img)
        bbox = torch.FloatTensor(bbox)
        return img, bbox, img_label
    
    def __len__(self):
        return len(self.samples)
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def get_dataloader():
    set_seed(209)
    root_path = '/data/enhance_imagenet/Imagenet/val'
    xmls_path = '/data/val_sigma_357911/test_xml/'   
    transform_test = transforms.Compose([transforms.Resize((224,224)), 
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ds = Dataset_obj(root_path=root_path, xmls_path=xmls_path, transform=transform_test)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4, shuffle=True)
    return dl


        
        
        



























# root_path = '/data/enhance_imagenet/Imagenet/val'
# xmls_path = '/data/val_sigma_357911/test_xml/'


# samples = get_ImgLabelXml(root_path=root_path, xmls_path=xmls_path)
# print(len(samples))

# f = open('imagenet_label.json')
# iii = json.load(f)

# for i in range(20):
#     idx = random.randint(0, len(samples)-1)
#     img_path, xml_path, label = samples[idx]
#     print(img_path)
#     print(xml_path)
#     sample = img_path.split('/')[5]
#     print(iii[sample][0])
#     print(label)
        