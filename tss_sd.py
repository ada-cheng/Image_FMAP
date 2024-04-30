import os
import torch
import argparse
from glob import glob
from PIL import Image
from utils.util_preprocess import resize_
from utils.utils_correspondence import co_pca
from extractor_sd import load_model, process_features_and_mask, get_mask
import torch.nn.functional as F



def extract_feat_and_mask_tss(model,aug,dir_path, save_path, save_path_name = None, facet='token', layer=11,category = 'bicycle', mask = True):
  

  device = 'cuda'
  
  if mask == False:
    save_path += '_nomask'
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  img_path_1 = f'{dir_path}/image1.png'
  img_path_2 = f'{dir_path}/image2.png'

  
  img_size = 840
  real_size = 960

  stride = 14
  
  category = category
  
  patch_size = 14
  num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)
  
  img1 = Image.open(img_path_1).convert('RGB')
  img1_dino,input_w_1 , input_h_1 = resize_(img1, img_size, resize=True, to_pil=True, edge=False)
  w1,h1 = img1.size
  if w1 > h1:
    targetw1 = 960
    targeth1 = 960 * h1 // w1
    img1 = img1.resize((targetw1,targeth1))
  else:
    targeth1 = 960
    targetw1 = 960 * w1 // h1
    img1 = img1.resize((targetw1,targeth1))

  img1_input_ = img1

  img2 = Image.open(img_path_2).convert('RGB')
  img2_dino,input_w_2 , input_h_2 = resize_(img2, img_size, resize=True, to_pil=True, edge=False)
  w2,h2 = img2.size
  if w2 > h2:
    targetw2 = 960
    targeth2 = 960 * h2 // w2
    img2 = img2.resize((targetw2,targeth2))
  else:
    targeth2 = 960
    targetw2 = 960 * w2 // h2
    img2 = img2.resize((targetw2,targeth2))
  img2_input_ = img2  
  input_text = None

  with torch.no_grad():
    features1 = process_features_and_mask(model_sd, aug, img1_input_, input_text=input_text,  mask=False, raw=True)
    features2 = process_features_and_mask(model_sd, aug, img2_input_, input_text=input_text,  mask=False, raw=True)
        
    processed_features1, processed_features2 = co_pca(features1, features2, PCA_DIMS)

    h1,w1 = processed_features1.shape[2:]
    h2,w2 = processed_features2.shape[2:]
    img1_desc = processed_features1.reshape(1, 1, -1, h1*w1).permute(0,1,3,2)
    img2_desc = processed_features2.reshape(1, 1, -1, h2*w2).permute(0,1,3,2)

            
    img1_desc_ = img1_desc.reshape((h1,w1,768)).permute(2,0,1)
    img2_desc_ = img2_desc.reshape((h2,w2,768)).permute(2,0,1)

    target_h1 = input_h_1 // 14 *14 //14
    target_w1 = input_w_1 // 14 *14 //14
    target_h2 = input_h_2 // 14 *14 //14
    target_w2 = input_w_2 // 14 *14 //14
            
    if img1_desc_.shape[1] != target_h1 or img1_desc_.shape[2] != target_w1:
             

        img1_desc_ = F.interpolate(img1_desc_.unsqueeze(0), size=(target_h1, target_w1), mode='bilinear', align_corners=False).squeeze(0)
    if img2_desc_.shape[1] != target_h2 or img2_desc_.shape[2] != target_w2:

        img2_desc_ = F.interpolate(img2_desc_.unsqueeze(0), size=(target_h2, target_w2), mode='bilinear', align_corners=False).squeeze(0)
            
    mask1 = get_mask(model_sd, aug, img1, category)
    mask2 = get_mask(model_sd, aug, img2, category)
    
    torch.save(mask1, f'{save_path}/{save_path_name}_mask1.pt')
    torch.save(mask2, f'{save_path}/{save_path_name}_mask2.pt')
    if mask:
      mask1_resized  = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(target_h1, target_w1), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
      mask2_resized  = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(target_h2, target_w2), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    
      img1_desc_ = img1_desc_ * mask1_resized.unsqueeze(0)
      img2_desc_ = img2_desc_ * mask2_resized.unsqueeze(0)
    
      torch.save(img1_desc_, f'{save_path}/{save_path_name}_img1.pt')
      torch.save(img2_desc_, f'{save_path}/{save_path_name}_img2.pt')
    else:

      img1_desc_ = F.interpolate(img1_desc_.unsqueeze(0), size=(target_h1, target_w1), mode='bilinear', align_corners=False).squeeze(0)
      img2_desc_ = F.interpolate(img2_desc_.unsqueeze(0), size=(target_h2, target_w2), mode='bilinear', align_corners=False).squeeze(0)
      torch.save(img1_desc_, f'{save_path}/{save_path_name}_img1.pt')
      torch.save(img2_desc_, f'{save_path}/{save_path_name}_img2.pt')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PASCAL', help='dataset name')
    parser.add_argument('--root_dir_path', type=str, default='/home/xinle/fmlib/data-tss/.s/TSS_CVPR2016/PASCAL', help='root directory path')
    parser.add_argument('--save_path', type=str, default='/home/xinle/fmlib/data-tss/TSS_CVPR2016/PASCAL_sd', help='save path')
    
    args = parser.parse_args()
    
    dataset_name = args.dataset
    root_dir_path = args.root_dir_path
    dir_list = sorted(glob(f'{root_dir_path}/*'))
    
    facet = 'token'
    layer = 11
    VER = "v1-5"
    SIZE = 960
    TIMESTEP = 100
    INDICES = [2,5,8,11]
    PCA_DIMS = [256,256,256]
    device = 'cuda'
  
    model_sd, aug = load_model(diffusion_ver=VER, image_size=SIZE, num_timesteps=TIMESTEP, block_indices=tuple(INDICES), decoder_only=False)
    model_sd = model_sd.to(device)
    
    for dir_path in dir_list[:1]:
      save_path_name = dir_path.split('/')[-1]

      save_path = args.save_path
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      if dataset_name == 'PASCAL':
        category = save_path_name.split('_')[0]
        extract_feat_and_mask_tss(model_sd,aug,dir_path, save_path, save_path_name, facet = facet, layer = layer, category = category,mask = False)
      elif dataset_name == 'FG3DCar':
        category = 'car'
        extract_feat_and_mask_tss(model_sd,aug,dir_path, save_path, save_path_name, facet = facet, layer = layer, category = category,mask = False)
      else:
        category = save_path_name.split('_')[0]
        extract_feat_and_mask_tss(model_sd,aug,dir_path, save_path, save_path_name, facet = facet, layer = layer, category = category,mask = False)
