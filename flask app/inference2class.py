import torch
import numpy as np
import cv2
import os 
import pandas as pd
import albumentations as A
#from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from albumentations.pytorch.functional import img_to_tensor
#from tensorboardX import SummaryWriter
from tqdm.notebook import tqdm
from helpers.config import load_config
import segmentation_models_pytorch as smp

# def convert_mask(mask):
#
#     color_mappings = {
#         0: (0, 0, 0),
#         1: (0, 255, 255),
#         2: (255, 0, 0),
#         3: (153, 76, 0),
#         4: (0, 153, 0)
#     }
#
#
#     height, width= mask.shape
#     rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
#
#
#     for class_idx, color in color_mappings.items():
#         class_mask = (mask == class_idx)
#         rgb_mask[class_mask] = color
#
#
#     return rgb_mask



class OilSegmentationModel:
    def __init__(self, transforms=None, conf=None, ckpt=None):
        self.conf = load_config(conf)
        self.ckpt = ckpt
        
        model = smp.Unet(encoder_name='efficientnet-b7', classes=2, encoder_weights=None)
        self.normalize = self.conf["input"]["normalize"]
        self.transforms = transforms
        checkpoint = torch.load(self.ckpt, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        #state_dict = {k[7:]: w for k, w in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        #model = model.cuda()
        self.model = model.eval()

    def segment_single_image(self, image, mode='path'):
        
        if mode == 'path':
            image = cv2.imread(image)
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        
        # Apply augmentations
        sample = self.transforms(image=image)

        sample['image'] = img_to_tensor(np.ascontiguousarray(sample['image']), self.normalize)
        
        with torch.no_grad():
            imgs = torch.unsqueeze(sample["image"], 0).float() #.cuda()
            output = self.model(imgs)
            #print(output.shape)
            pred = torch.softmax(output, dim=1)
            #print(pred.shape)
            mask = pred.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float)
            #argmax = torch.argmax(pred, dim=1)
            #print(argmax.shape)
            #mask = argmax.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            #print(np.unique(mask))

        #mask = size(mask, (image_to_segment.shape[:2]))
        #mask = np.rot90(mask)
        #mask = cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
        #print(np.unique(mask))
        #final_mask = np.where(mask==16, 255, mask)
        

        return mask #pred

def prepare_segment():

    conf = load_config('helpers/effb7_conf.json')

    transform = A.Compose([
            A.Crop(0,0,640,640, p=1.0)
        ])

    conf_path = 'helpers/effb7_conf.json'
    ckpt_path = 'segment_efficientnet-b7_efficientnet-b7_best_miou.pt'

    segmenter = OilSegmentationModel(transform, conf_path, ckpt_path)

    return segmenter



def segment(img):
    #img = cv2.imread(f'Preprocessing Folder/png_files/{png_name}')

    #img = cv2.medianBlur(img, 3)

    segmenter = prepare_segment()

    # Define the patch size and overlap
    patch_size = (640, 640)
    overlap = 0.5
    #print(img.shape)
    # Load the semantic segmentation model
    segmentation_map = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
    #print(segmentation_map.shape)
    # Loop over the image patches
    for y in range(0, img.shape[0], int(patch_size[0] * overlap)):
        for x in range(0, img.shape[1], int(patch_size[1] * overlap)):



            patch = img[y:y+patch_size[0], x:x+patch_size[1]]
            #print(patch.shape)
            #print(patch.shape[0])

            if patch.shape[1] < patch_size[1] and patch.shape[0] == patch_size[0]:
                patch = img[y:y+patch_size[0], img.shape[1]-patch_size[1]:img.shape[1]]
                patch_seg_map = segmenter.segment_single_image(patch, 'img')
                segmentation_map[y:y+patch_size[0], img.shape[1]-patch_size[1]:img.shape[1]] += patch_seg_map
                #print(segmentation_map.shape)
            elif patch.shape[0] < patch_size[0] and patch.shape[1] == patch_size[1]:
                patch = img[img.shape[0]-patch_size[0]:img.shape[0], x:x+patch_size[1]]
                patch_seg_map = segmenter.segment_single_image(patch, 'img')
                segmentation_map[img.shape[0]-patch_size[0]:img.shape[0], x:x+patch_size[1]] += patch_seg_map
            elif patch.shape[0] < patch_size[0] and patch.shape[1] < patch_size[1]:
                patch = img[img.shape[0]-patch_size[0]:img.shape[0], img.shape[1]-patch_size[1]:img.shape[1]]
                patch_seg_map = segmenter.segment_single_image(patch, 'img')
                segmentation_map[img.shape[0]-patch_size[0]:img.shape[0], img.shape[1]-patch_size[1]:img.shape[1]] += patch_seg_map
            else:
            #patch = cv2.resize(patch, (256, 256))
            #patch = np.expand_dims(patch, axis=0)

                patch_seg_map = segmenter.segment_single_image(patch, 'img')
            #patch_seg_map = cv2.resize(patch_seg_map, patch_size)
                segmentation_map[y:y+patch_size[0], x:x+patch_size[1]] += patch_seg_map
                
    segmentation_map = np.argmax(segmentation_map, axis=-1)
    segmentation_map *= 255
    
    print(type(segmentation_map))

    return segmentation_map




