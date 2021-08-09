import torch
from utils.dataloader import get_dataloader, set_seed
from tqdm import tqdm
from utils.utils import save_headmp, compute_iou, compute_EBPG
import numpy as np
import sys

from methods.GradCam import GradCam
from methods.GradCampp import GradCampp
from methods.LoG import LoG
    
    
def test_IoU(model, device, method, percent):
    dataloader = get_dataloader()
    
    model.eval()
        
    total_sample = 0
    total_IoU = 0
    
    if method == 'gradcam':
        me = GradCam(model, percent, device)
    elif method == 'gradcam++':
        me = GradCampp(model, percent, device)
    elif method == 'log':
        me = LoG(model, percent, device)
    else:
        me = None
    
    with tqdm(total=len(dataloader), 
              desc=f'Total Samples: {total_sample}', file=sys.stdout) as pbar:
        for idx, (inputs, bbox, label) in enumerate(dataloader):
        
            bbox = bbox[0]
            if len(bbox) != 4:
                continue
            if ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])).item() / (224 * 224) > 0.5:
                continue
            
            b_box, sm = me.get_bbox(inputs, label)
            
            if b_box is None or sm is None:
                continue
                
            total_sample += 1
            total_IoU += compute_iou(bbox, b_box)
            
            pbar.set_description(desc=f'Total Samples: {total_sample} | ' + \
                                 f'Mean IoU: {total_IoU / (total_sample):6.4f} | ',
                                 refresh=True)
            pbar.update()
    
    print('Total samples: {}'.format(total_sample))
    print('Mean IoUG: {}'.format(total_IoU / total_sample))

            
            
        

