Saliency Map Generation through Lateral Inhibition Mechanism
==================
This repository provides code to replicate the paper
**Saliency Map Generation through Lateral Inhibition Mechanism**
You can use the instructions below to setup an environment with the right dependencies.
```
pip install -r requirements.txt
```
### Test IoU
Please change the imagenet data path (images and annotations) in utils.dataloader file.
```
python main.py -dv your device -mtd method -cpt your resnet50 pretrained model
```
