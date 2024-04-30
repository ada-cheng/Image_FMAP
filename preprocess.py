import os
import torch
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import argparse
import time
from glob import glob
import json
from PIL import Image
from utils.util_preprocess import resize_, foreground_preprocess
from utils.utils_correspondence import co_pca, generate_cute_rainbow
from extractor import ViTExtractor
from extractor_sd import load_model, process_features_and_mask, get_mask
import matplotlib.pyplot as plt
from pca_visualizer import pca_
import torch.nn.functional as F



def extract_feat_and_mask_tss(dir_path, save_path, save_path_name = None, facet='token', layer=11,dataset_name = 'PASCAL'):
  
  
  img_path_1 = f'{dir_path}/image1.png'
  img_path_2 = f'{dir_path}/image2.png'

  
  img_size = 840
  model_type = 'dinov2_vitb14'
  stride = 14
  device = 'cuda'
  
  img1 = Image.open(img_path_1).convert('RGB')
  h,w = img1.size
  img1,w1,h1 = resize_(img1, img_size, resize=True, to_pil=False, edge=False)
  if w1 > h1 :
    img1_ = img1[(w1-h1)//2:(w1-h1)//2+h1,:]
  else:
    img1_ = img1[:,(h1-w1)//2:(h1-w1)//2+w1]
  img1 = Image.fromarray(img1_)
        
  img1 = img1.resize((w1//14*14,h1//14*14))
  img2 = Image.open(img_path_2).convert('RGB')
  h,w = img2.size
  img2,w2,h2 = resize_(img2, img_size, resize=True, to_pil=False, edge=False)
  if w2 > h2 :
    img2_ = img2[(w2-h2)//2:(w2-h2)//2+h2,:]
  else:
    img2_ = img2[:,(h2-w2)//2:(h2-w2)//2+w2]
  img2 = Image.fromarray(img2_)
        
  img2 = img2.resize((w2//14*14,h2//14*14))

  extractor = ViTExtractor(model_type=model_type, device=device, stride=stride)

  with torch.no_grad():
    img1_desc= extractor.preprocess_pil(img1)
    img1_desc = extractor.extract_descriptors(img1_desc.to(device), layer=layer, facet=facet)
    h,w = img1.size[1] // stride, img1.size[0] // stride
    img1_desc = img1_desc.reshape((h,w,-1)).permute(2,0,1)

    torch.save(img1_desc, f'{save_path}/img1.pt') if save_path_name is None else torch.save(img1_desc, f'{save_path}/{save_path_name}_img1.pt')

    img2_desc= extractor.preprocess_pil(img2)
    img2_desc = extractor.extract_descriptors(img2_desc.to(device), layer=layer, facet=facet)
    h,w = img2.size[1] // stride, img2.size[0] // stride
    img2_desc = img2_desc.reshape((h,w,-1)).permute(2,0,1)

    torch.save(img2_desc, f'{save_path}/img2.pt') if save_path_name is None else torch.save(img2_desc, f'{save_path}/{save_path_name}_img2.pt')

  return img1_desc,img2_desc
    
  




if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='PASCAL')
    parser.add_argument('--root_dir_path', type=str, default='/home/xinle/fmlib/data-tss/.s/TSS_CVPR2016/PASCAL')
    parser.add_argument('--facet', type=str, default='token')
    parser.add_argument('--layer', type=int, default=11)
    parser.add_argument('--save_path', type=str, default='/home/xinle/fmlib/data-tss/TSS_CVPR2016/PASCAL_feat_token_11_NOMASK')
    parser.add_argument('--root_path', type=str, default='/home/xinle/fmlib/data-tss/.s/TSS_CVPR2016/')
    
    args = parser.parse_args()
    dataset_name = args.dataset_name
    root_dir_path = f'{args.root_path}{dataset_name}'
    dir_list = sorted(glob(f'{root_dir_path}/*'))
    facet = args.facet
    layer = args.layer
    for dir_path in dir_list:
      save_path_name = dir_path.split('/')[-1]
      save_path = args.save_path
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      extract_feat_and_mask_tss(dir_path, save_path, save_path_name, facet = facet, layer = layer, dataset_name=dataset_name)
