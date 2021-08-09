import torch
import argparse
import torchvision.models as models
from tester import test_IoU
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('-dv', '--device', default='cuda:0')
parser.add_argument('-mtd', '--method', default='gradcam', choices={'gradcam', 'gradcam++', 'log'})
parser.add_argument('-cpt', default='')
parser.add_argument('--percent', default=0.15, type=float)

if __name__ == '__main__':
    parse_args = parser.parse_args()
    
    model = models.resnet50(pretrained=False)
    dicts = torch.load(parse_args.cpt)

    new_dicts = {}
    for key in dicts.keys():
        new_dicts[key.replace('module.','')] = dicts[key]
    model.load_state_dict(new_dicts)
    
    test_IoU(model, parse_args.device, parse_args.method, parse_args.percent)
    
    
        