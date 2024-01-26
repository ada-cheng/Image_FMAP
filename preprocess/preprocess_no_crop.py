import torch
from tqdm import tqdm
import numpy as np
import os
from typing import List, Tuple
from loguru import logger
from dino import Dino_rgb_extractor
import argparse
import time
from glob import glob
import json
from PIL import Image, ImageDraw
from utils.util_preprocess import resize_, foreground_preprocess
from extractor import ViTExtractor
import matplotlib.pyplot as plt
import torch.nn.functional as F


def set_pixels_outside_box(img, bbox):
    # Create a binary mask
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)

    # Apply the mask to the original image
    result_img = Image.new('RGB', img.size, (0, 0, 0))
    result_img.paste(img, mask=mask)

    return result_img


def load_spair_data(path, size=256, category='cat', split='test', subsample=None):
    np.random.seed(42)
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*:{category}.json'))
    if subsample is not None and subsample > 0:
        pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]

    pairs = pairs[:subsample] if subsample is not None else pairs

    files = []
    bboxes = []
    for pair in pairs:
        with open(pair) as f:
            data = json.load(f)
        bbox_src = data["src_bndbox"]
        bbox_tgt = data["trg_bndbox"]
        bboxes.append(bbox_src)
        bboxes.append(bbox_tgt)
        assert category == data["category"]
        assert data["mirror"] == 0
        source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
        target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'
        files.append(source_fn)
        files.append(target_fn)
    return files,bboxes

def extract_pair_features(model,save_path, files, category, layer=11, img_size=840, facet='token', bboxes=None):    
    img_size = 840
  
    if 'l' in model:
        layer = 23
    elif 'g' in model:
        layer = 39
    stride = 14
    device = 'cuda'
    extractor = ViTExtractor(model_type=model, device=device, stride=stride)
    N = len(files) // 2
    if not os.path.exists(f'{save_path}/{model}_{category}_{layer}_{facet}_np'):
        os.makedirs(f'{save_path}/{model}_{category}_{layer}_{facet}_np')
    for pair_idx in range(N):
        # Load image 1
        image_path = files[2*pair_idx]

        img1 = Image.open(image_path).convert('RGB')
        h,w = img1.size
        bbox1 = bboxes[2*pair_idx]

        w1_ori,h1_ori = img1.size
        img1,w1,h1 = resize_(img1, img_size, resize=True, to_pil=False, edge=False)
        scale_factor = w1_ori / w1
        bbox1 = [int(x / scale_factor) for x in bbox1]
        if w1 > h1 :
            img1_ = img1[(w1-h1)//2:(w1-h1)//2+h1,:]
        else:
            img1_ = img1[:,(h1-w1)//2:(h1-w1)//2+w1]
        img1 = Image.fromarray(img1_)
        img1 = img1.resize((w1//14*14,h1//14*14))

         
        image_path = files[2*pair_idx + 1]

        img2 = Image.open(image_path).convert('RGB')
        bbox2 = bboxes[2*pair_idx + 1]
        w2_ori,h2_ori = img2.size
        img2,w2,h2 = resize_(img2, img_size, resize=True, to_pil=False, edge=False)
        scale_factor = w2_ori / w2
        bbox2 = [int(x / scale_factor) for x in bbox2]
        if w2 > h2 :
            img2_ = img2[(w2-h2)//2:(w2-h2)//2+h2,:]
        else:
            img2_ = img2[:,(h2-w2)//2:(h2-w2)//2+w2]
        img2 = Image.fromarray(img2_)
        
        img2 = img2.resize((w2//14*14,h2//14*14))

    
        with torch.no_grad():
          img1_desc= extractor.preprocess_pil(img1)
          img1_desc = extractor.extract_descriptors(img1_desc.to(device), layer=layer, facet=facet)
          h,w = img1.size[1] // stride, img1.size[0] // stride
          img1_desc = img1_desc.reshape((h,w,-1)).permute(2,0,1)

          torch.save(img1_desc, f'{save_path}/{model}_{category}_{layer}_{facet}_np/{pair_idx}_1.pt')
          img1 = torch.from_numpy(np.array(img1))
 
          img2_desc= extractor.preprocess_pil(img2)
          img2_desc = extractor.extract_descriptors(img2_desc.to(device), layer=layer, facet=facet)
          h,w = img2.size[1] // stride, img2.size[0] // stride
          img2_desc = img2_desc.reshape((h,w,-1)).permute(2,0,1)

          torch.save(img2_desc, f'{save_path}/{model}_{category}_{layer}_{facet}_np/{pair_idx}_2.pt')
       
     
       
          
        if pair_idx > 100:
            break
    



if __name__ == "__main__":
    data_dir = 'data/SPair-71k' 
    save_path = 'pair-feat/SPair-71k'
    model_type = 'dinov2_vitb14'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    categories = os.listdir(os.path.join(data_dir, 'ImageAnnotation'))
   
    for cat in categories:
        files,bboxes = load_spair_data(data_dir, size=840, category=cat, subsample=20)
        extract_pair_features(model_type, save_path, files, cat,layer=9,facet = 'token', bboxes=bboxes)
        