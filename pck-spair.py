import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple
from matplotlib.patches import ConnectionPatch
from loguru import logger
import argparse
import time
from glob import glob
import json
import torch.nn.functional as F
from matplotlib import gridspec


EPS = 1e-8
def resize(img, target_res, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
          
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
       
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
            
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
      
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas,width,height

def pairwise_sim(x: torch.Tensor, y: torch.Tensor, p=2, normalize=False) -> torch.Tensor:
    # compute similarity based on euclidean distances
  
    if normalize:
        x = torch.nn.functional.normalize(x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)
    result_list=[]
    num_token_x = x.shape[2]
    
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)
        result_list.append(torch.nn.PairwiseDistance(p=p)(token, y)*(-1))
    return torch.stack(result_list, dim=2)

def draw_correspondences_lines(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]], 
                               gt_points2: List[Tuple[float, float]], image1: Image.Image, 
                               image2: Image.Image, threshold=None) -> plt.Figure:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param gt_points2: a list of ground truth (y, x) coordinates of image2.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :param threshold: distance threshold to determine correct matches.
    :return: a figure of images with marked points and lines between them showing correspondence.
    """

    points2=points2.cpu().numpy()
    gt_points2=gt_points2.cpu().numpy()
    # resize image to the same height
    height1 = image1.size[1]
    height2 = image2.size[1]
    target_height = 300
    scale_factor1 = target_height / height1
    scale_factor2 = target_height / height2
    resized_image1 = image1.resize((int(image1.size[0] * scale_factor1), target_height))
    points1 = [(int(scale_factor1 * x), int(scale_factor1 * y)) for (x, y) in points1]
    resized_image2 = image2.resize((int(image2.size[0] * scale_factor2), target_height))
    points2 = [(int(scale_factor2 * x), int(scale_factor2 * y)) for (x, y) in points2]
    gt_points2 = [(int(scale_factor2 * x), int(scale_factor2 * y)) for (x, y) in gt_points2]
    gt_points2 = [(float(x), float(y)) for (x, y) in gt_points2]
    image1 = resized_image1
    image2 = resized_image2
 
    def compute_correct():
        alpha = torch.tensor([0.1, 0.05, 0.01,0.15])
        correct = torch.zeros(len(alpha))
        err = (torch.tensor(points2) - torch.tensor(gt_points2)).norm(dim=-1)
        err = err.unsqueeze(0).repeat(len(alpha), 1)
        correct = err < threshold.unsqueeze(-1) if len(threshold.shape)==1 else err < threshold
        return correct

    correct = compute_correct()[0]
   

    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)

    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                            "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 0.03*max(image1.size), 0.01*max(image1.size)
    
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,8))
    ax1.imshow(image1)
    ax2.imshow(image2)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=0.1, hspace=0.1)
  
    x1_lim, y1_lim = ax1.get_xlim(), ax1.get_ylim()

# Apply the limits to the second subplot
    ax2.set_xlim(x1_lim)
    ax2.set_ylim(y1_lim)
    ax1.axis('off')
    ax2.axis('off')

    for i, (point1, point2) in enumerate(zip(points1, points2)):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=colors[i], edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=colors[i], edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=colors[i], edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=colors[i], edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)

        # Draw lines
        color = 'blue' if correct[i].item() else 'red'
        con = ConnectionPatch(xyA=(x2, y2), xyB=(x1, y1), coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color=color, linewidth=1.5)
        ax2.add_artist(con)

    return fig

def draw_correspondences_gathered(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]],
                        image1: Image.Image, image2: Image.Image) -> plt.Figure:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: a figure of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)

    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                            "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 0.03*max(image1.size), 0.01*max(image1.size)
    
    # plot a subfigure put image1 in the top, image2 in the bottom
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)

    for point1, point2, color in zip(points1, points2, colors):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)

    return fig

def preprocess_kps_pad(kps, img_width, img_height, size_h, size_w):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = max(size_h / img_height, size_w / img_width)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size_w * img_height / img_width))
        offset_y = int((size_h - new_h) / 2)
        offset_x = 0
  
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size_h * img_width / img_height))
        offset_x = int((size_w - new_w) / 2)
        offset_y = 0
    
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    if not COUNT_INVIS:
        kps *= kps[:, 2:3].clone()  
    return kps, offset_x, offset_y, scale

def load_spair_data(path, size=256, category='cat', split='test', subsample=None):
    np.random.seed(SEED)
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*:{category}.json'))
    if subsample is not None and subsample > 0:
        pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
   
    logger.info(f'Number of SPairs for {category} = {len(pairs)}')
    files = []
    thresholds = []
    category_anno = list(glob(f'{path}/ImageAnnotation/{category}/*.json'))[0]
    with open(category_anno) as f:
        num_kps = len(json.load(f)['kps'])
    logger.info(f'Number of SPair key points for {category} <= {num_kps}')
    kps = []
    blank_kps = torch.zeros(num_kps, 3)
    bboxes = []
    for pair in pairs:
        with open(pair) as f:
            data = json.load(f)
        assert category == data["category"]
        assert data["mirror"] == 0
        source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
        target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'

        source_bbox = np.asarray(data["src_bndbox"])
        target_bbox = np.asarray(data["trg_bndbox"])
        bboxes.append(source_bbox)
        bboxes.append(target_bbox)


        source_size = data["src_imsize"][:2]  # (W, H)
        target_size = data["trg_imsize"][:2]  # (W, H)

        kp_ixs = torch.tensor([int(id) for id in data["kps_ids"]]).view(-1, 1).repeat(1, 3)
        source_raw_kps = torch.cat([torch.tensor(data["src_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        source_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=source_raw_kps)
        size_h, size_w = 0,0
        if source_size[0] < source_size[1]:
            size_h = size
            size_w = int(np.around(size * source_size[0] / source_size[1]))
        else:
            size_w = size
            size_h = int(np.around(size * source_size[1] / source_size[0]))
            
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size_h, size_w)
        
        target_raw_kps = torch.cat([torch.tensor(data["trg_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        target_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=target_raw_kps)
        if target_size[0] < target_size[1]:
            size_h = size
            size_w = int(np.around(size * target_size[0] / target_size[1]))
        else:
            size_w = size
            size_h = int(np.around(size * target_size[1] / target_size[0]))
        
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size_h, size_w)

        thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0])*trg_scale)

        kps.append(source_kps)
        kps.append(target_kps)
        files.append(source_fn)
        files.append(target_fn)

    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    logger.info(f'Final number of used key points: {kps.size(1)}')
    return files, kps, thresholds,bboxes

def load_files(base_file_name,number,suffix):
    files = []
    for i in range(number):
        files.append(base_file_name+str(i)+suffix)
    return files


def compute_pck(model,save_path, files, kps, category, dist='cos', thresholds=None, facet= 'token', bbox = None):

    img_size = 840
    
    stride = 14 
    device ='cuda'
    
    patch_size = 14

    if not os.path.exists(f'{save_path}/{category}'):
        os.makedirs(f'{save_path}/{category}')
        
    current_save_results = 0
    gt_correspondences = []
    pred_correspondences = []
    if thresholds is not None:
        thresholds = torch.tensor(thresholds).to(device)
        bbox_size=[]
    N = len(files) // 2
    
    pbar = tqdm(total=N)

    prefix = PATH_TO_FMAP
    suffix = '.pt'
    fmap_files = load_files(prefix, 20, suffix)

    prefix_sd_1 = PATH_TO_SD
    suffix_sd_1 = '_1.pt'
    sd_files_1 = load_files(prefix_sd_1, 20, suffix_sd_1)
    
    prefix_sd_2 = PATH_TO_SD
    suffix_sd_2 = '_2.pt'
    sd_files_2 = load_files(prefix_sd_2, 20, suffix_sd_2)
    
    prefix_dino_1 = PATH_TO_DINO
    suffix_dino_1 = '_1.pt'
    dino_files_1 = load_files(prefix_dino_1, 20, suffix_dino_1)
    
    prefix_dino_2 = PATH_TO_DINO
    suffix_dino_2 = '_2.pt'
    dino_files_2 = load_files(prefix_dino_2, 20, suffix_dino_2)
 
    prefix_dino_val_1 = PATH_TO_DINO_VAL
    suffix_dino_val_1 = '_1.pt'
    dino_val_files_1 = load_files(prefix_dino_val_1, 20, suffix_dino_val_1)
    
    prefix_dino_val_2 = PATH_TO_DINO_VAL
    suffix_dino_val_2 = '_2.pt'
    dino_val_files_2 = load_files(prefix_dino_val_2, 20, suffix_dino_val_2)
    
    
    prefix_mask_1 = PATH_TO_MASK
    suffix_mask_1 = '_1_mask.pt'
    mask_files_1 = load_files(prefix_mask_1, 20, suffix_mask_1)
    
    prefix_mask_2 = PATH_TO_MASK
    suffix_mask_2 = '_2_mask.pt'
    mask_files_2 = load_files(prefix_mask_2, 20, suffix_mask_2)
    
    prefix_crop = PATH_TO_CROP
    suffx_crop_1 = '_1.pt'
    crop_files_1 = load_files(prefix_crop, 20, suffx_crop_1)
    suffix_crop_2 = '_2.pt'
    crop_files_2 = load_files(prefix_crop, 20, suffix_crop_2)
    

    for pair_idx in range(N):
    
        # Load image 1
        img1 = Image.open(files[2*pair_idx]).convert('RGB')
        w_ori1,h_ori1 = img1.size
        bbox1 = bbox[2*pair_idx]
        
        img1,width1,height1 = resize(img1, img_size, resize=True, to_pil=False, edge=False)
        scale_factor = w_ori1 / width1
        bbox1 = [int(x / scale_factor) for x in bbox1]
        if width1 < height1:
            img1_ = img1[:,(height1-width1)//2:(height1-width1)//2+width1]
        else:
            img1_ = img1[(width1 - height1)//2:(width1 - height1)//2+height1]
        img1 = Image.fromarray(img1_)
      
        
        img1 = img1.resize((width1//14*14,height1//14*14))
        width1,height1 = img1.size
        img1_kps = kps[2*pair_idx]

        # Get patch index for the keypoints
        img1_y, img1_x = img1_kps[:, 1].numpy(), img1_kps[:, 0].numpy() # y is width, x is height
        
        num_patches_y = int(patch_size / stride * (width1 // patch_size - 1) + 1)
        num_patches_x = int(patch_size / stride * (height1 // patch_size - 1) + 1)
        
        img1_y_patch = (num_patches_y / width1 * img1_y).astype(np.int32)
        img1_x_patch = (num_patches_x / height1 * img1_x).astype(np.int32)
        img1_y_patch[img1_y_patch >= num_patches_x] = num_patches_x - 1
        img1_patch_idx = num_patches_y * img1_y_patch + img1_x_patch


        img2 = Image.open(files[2*pair_idx+1]).convert('RGB')
        w_ori2,h_ori2 = img2.size
        bbox2 = bbox[2*pair_idx+1]

        
        img2,width2,height2 = resize(img2, img_size, resize=True, to_pil=False, edge=False)
        scale_factor = w_ori2 / width2
        bbox2 = [int(x / scale_factor) for x in bbox2]
        if width2 < height2:
            img2 = img2[:,(height2-width2)//2:(height2-width2)//2+width2]
        else:
            img2 = img2[(width2 - height2)//2:(width2 - height2)//2+height2]
        img2 = Image.fromarray(img2)
        img2 = img2.resize((width2//14*14,height2//14*14))
        width2,height2 = img2.size
        img2_kps = kps[2*pair_idx+1]
       

        # Get patch index for the keypoints
        img2_y, img2_x = img2_kps[:, 1].numpy(), img2_kps[:, 0].numpy()
        num_patches_y_2 = int(patch_size / stride * (width2 // patch_size - 1) + 1)
        num_patches_x_2 = int(patch_size / stride * (height2 // patch_size - 1) + 1)

        with torch.no_grad():
            
     
          if SD and PREVFMAP:
              pass
          else:

            h,w = height1 // stride, width1 // stride
      
            img1_desc = torch.load(dino_files_1[pair_idx]).to(device)
            torch.save(img1_desc, f'{save_path}/{category}/{pair_idx}_img1.pt')   
            img1_desc = img1_desc.reshape((img1_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2) 
            h,w = height2 // stride, width2 // stride

            img2_desc = torch.load(dino_files_2[pair_idx]).to(device) 
            img2_desc = img2_desc.reshape((img2_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2) 
          
        
          
          if SD:
              h_sd1,w_sd1 = None, None
              h_sd2,w_sd2 = None, None
              img1_desc_sd = torch.load(sd_files_1[pair_idx]).to(device)
              if img1_desc_sd.shape[1] != height1 // stride or img1_desc_sd.shape[2] != width1 // stride:
                  h_sd1, w_sd1 = img1_desc_sd.shape[1], img1_desc_sd.shape[2]
                  img1_desc_sd = F.interpolate(img1_desc_sd.unsqueeze(0), size=(height1 // stride, width1 // stride), mode='bilinear', align_corners=False).squeeze(0)
              img1_desc_sd = img1_desc_sd.reshape((img1_desc_sd.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2)
              
              img2_desc_sd = torch.load(sd_files_2[pair_idx]).to(device)
              if img2_desc_sd.shape[1] != height2 // stride or img2_desc_sd.shape[2] != width2 // stride:
                  h_sd2, w_sd2 = img2_desc_sd.shape[1], img2_desc_sd.shape[2]
                  img2_desc_sd = F.interpolate(img2_desc_sd.unsqueeze(0), size=(height2 // stride, width2 // stride), mode='bilinear', align_corners=False).squeeze(0)
              img2_desc_sd = img2_desc_sd.reshape((img2_desc_sd.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2)

              
          if VAL:
          
              h,w = height1 // stride, width1 // stride
              img1_desc_val = img1_desc_val.reshape((h,w,-1)).permute(2,0,1)
              img1_desc_val = torch.load(dino_val_files_1[pair_idx]).to(device)
              img1_desc_val = img1_desc_val.reshape((img1_desc_val.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2) # (1, 1, h*w, dim)
           
              h, w = height2 // stride, width2 // stride
              img2_desc_val = img2_desc_val.reshape((h,w,-1)).permute(2,0,1)
              img2_desc_val = torch.load(dino_val_files_2[pair_idx]).to(device)
              img2_desc_val = img2_desc_val.reshape((img2_desc_val.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2) # (1, 1, h*w, dim)
              

          if SD and not CAT and not VAL and not PREVFMAP:
              img1_desc = img1_desc_sd
              img2_desc = img2_desc_sd

          if CAT and SD and not VAL and not PREVFMAP:

              img1_desc = img1_desc / (img1_desc.norm(dim=-1, keepdim=True) )
              img2_desc = img2_desc / (img2_desc.norm(dim=-1, keepdim=True) )

              img1_desc_sd = img1_desc_sd / (img1_desc_sd.norm(dim=-1, keepdim=True) +EPS )
              img2_desc_sd = img2_desc_sd / (img2_desc_sd.norm(dim=-1, keepdim=True) +EPS)
             
              img1_desc = torch.cat((img1_desc_sd, img1_desc), dim=-1)
              img2_desc = torch.cat((img2_desc_sd, img2_desc), dim=-1)
          if CAT and VAL and not SD and not PREVFMAP:
           
              img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
              img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
              img1_desc_val = img1_desc_val / img1_desc_val.norm(dim=-1, keepdim=True)
              img2_desc_val = img2_desc_val / img2_desc_val.norm(dim=-1, keepdim=True)
              img1_desc = torch.cat((img1_desc,img1_desc_val), dim=-1)
              img2_desc = torch.cat((img2_desc,img2_desc_val), dim=-1)
          if VAL and not CAT and not SD and not PREVFMAP:
              img1_desc = img1_desc_val
              img2_desc = img2_desc_val


          if PREVFMAP and not CAT and SD:
              img1_desc = img1_desc_sd
              img2_desc = torch.load(fmap_files[pair_idx]).permute(2,0,1).to(device)
              if img2_desc.shape[1] != height2 // stride or img2_desc.shape[2] != width2 // stride:
                  img2_desc = F.interpolate(img2_desc.unsqueeze(0), size=(height2 // stride, width2 // stride), mode='bilinear', align_corners=False).squeeze(0)
              img2_desc = img2_desc.reshape((img2_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2) 
              img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
              img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
        
          if PREVFMAP and not CAT and not SD and not CROP:

              mask1 = torch.load(mask_files_1[pair_idx]).to(device)
              mask2 = torch.load(mask_files_2[pair_idx]).to(device)
              mask1 = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(height1 // stride, width1 // stride), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
              mask2 = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(height2 // stride, width2 // stride), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
              img2_desc_ = torch.load(fmap_files[pair_idx],map_location=device)
              img2_desc = img2_desc.reshape((height2 // stride, width2 // stride, -1))

              img2_desc = img2_desc_
              img2_desc = img2_desc.permute(2,0,1).to(device)
              img1_desc = img1_desc.reshape((height1// stride, width1 // stride, -1))

              img1_desc = img1_desc.permute(2,0,1).to(device)
              img1_desc = img1_desc.reshape((img1_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2)

              img2_desc = img2_desc.reshape((img2_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2)
          if PREVFMAP and not CAT and not SD and CROP:
              img2_desc_crop = torch.load(fmap_files[pair_idx]).permute(2,0,1).to(device)
              img1_desc_crop = torch.load(crop_files_1[pair_idx])
          
              bbox1 = [int(x / stride) for x in bbox1]
              bbox2 = [int(x / stride) for x in bbox2]
              tgt_h = bbox2[3] - bbox2[1]
              tgt_w = bbox2[2] - bbox2[0]

              img2_desc_crop = F.interpolate(img2_desc_crop.unsqueeze(0), size=(tgt_h,tgt_w), mode='bilinear', align_corners=False).squeeze(0)
              img2_desc_crop = img2_desc_crop.permute(1,2,0)
              
              img2_desc = img2_desc.reshape((height2 // stride, width2 // stride, -1))
        
              img2_desc[bbox2[1]:bbox2[3],bbox2[0]:bbox2[2]] = img2_desc_crop[:,:,:]
              img2_desc = img2_desc.permute(2,0,1).to(device)
              
              img2_desc = img2_desc.reshape((img2_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2)

              tgt_h = bbox1[3] - bbox1[1]
              tgt_w = bbox1[2] - bbox1[0]
             
              img1_desc_crop = F.interpolate(img1_desc_crop.unsqueeze(0), size=(tgt_h,tgt_w), mode='bilinear', align_corners=False).squeeze(0)
              img1_desc_crop = img1_desc_crop.permute(1,2,0)
              
              img1_desc = img1_desc.reshape((height1 // stride, width1 // stride, -1))
              img1_desc[bbox1[1]:bbox1[3],bbox1[0]:bbox1[2]] = img1_desc_crop[:,:,:]
              
              img1_desc = img1_desc.permute(2,0,1).to(device)
              img1_desc = img1_desc.reshape((img1_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2)

        vis = img1_kps[:, 2] * img2_kps[:, 2] > 0
        if COUNT_INVIS:
            vis = torch.ones_like(vis)
        mask1 = torch.load(mask_files_1[pair_idx]).to(device)
        img1_desc = img1_desc.reshape((height1 // stride, width1 // stride, -1)).permute(2,0,1)
        mask1 = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(height1 // stride, width1 // stride), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        if MASK:
            img1_desc[(mask1 == 0).unsqueeze(0).repeat(img1_desc.shape[0],1,1)] = 1e5
        img1_desc = img1_desc.reshape((img1_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2)
        
        mask2 = torch.load(mask_files_2[pair_idx]).to(device)
        
        img2_desc = img2_desc.reshape((height2 // stride, width2 // stride, -1)).permute(2,0,1)
        mask2 = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(height2 // stride, width2 // stride), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        
        if not PREVFMAP and MASK:
            img2_desc[(mask2 == 0).unsqueeze(0).repeat(img2_desc.shape[0],1,1)] = 1e5
            
        img2_desc = img2_desc.reshape((img2_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2)
        # Get similarity matrix
        if dist == 'l2':
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=2).squeeze()
        elif dist == 'l1':
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=1).squeeze()
        elif dist == 'l2_norm':
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=2, normalize=True).squeeze()
        elif dist == 'l1_norm':
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=1, normalize=True).squeeze()
        else:
            raise ValueError('Unknown distance metric')
      
        # Get nearest neighors if masked to smooth out outliers
        if MASK:
            for i in range(len(img1_patch_idx)):
                if mask1[img1_patch_idx[i] // num_patches_y, img1_patch_idx[i] % num_patches_y] == 0:
                # query nearest neighbor in mask
                    point_tensor = torch.tensor([img1_patch_idx[i] // num_patches_y, img1_patch_idx[i] % num_patches_y]).unsqueeze(0)
                    one_indices = torch.nonzero(mask1)
                    one_indices = one_indices.to(device)
                    point_tensor = point_tensor.to(device)
                    distance = torch.norm(one_indices.float() - point_tensor.float(), dim=1, p=2)
                    nearest_index = torch.argmin(distance).item()
                    nearest_1 = one_indices[nearest_index]
                    img1_patch_idx[i] = nearest_1[0] * num_patches_y + nearest_1[1]
               
     
     
        
        nn_1_to_2 = torch.argmax(sim_1_to_2[img1_patch_idx], dim=1)

        feat_sim_ans = sim_1_to_2[img1_patch_idx,nn_1_to_2]
        for i in range(len(img1_patch_idx)):
            if img1_patch_idx[i] == 0:
                feat_sim_ans[i] = 0
        feat_sim_ans = feat_sim_ans[feat_sim_ans.nonzero()]
        img1_patch_idx = img1_patch_idx[img1_patch_idx.nonzero()]

        nn_y_patch, nn_x_patch = nn_1_to_2 // num_patches_y_2, nn_1_to_2 % num_patches_y_2
        nn_x = (nn_x_patch - 1) * stride + stride + patch_size // 2 - .5
        nn_y = (nn_y_patch - 1) * stride + stride + patch_size // 2 - .5
        kps_1_to_2 = torch.stack([nn_x, nn_y]).permute(1, 0)
    
        gt_correspondences.append(img2_kps[vis][:, [1,0]])
        pred_correspondences.append(kps_1_to_2[vis][:, [1,0]])
        
        
        if thresholds is not None:
            bbox_size.append(thresholds[pair_idx].repeat(vis.sum()))
        
        if current_save_results!=TOTAL_SAVE_RESULT:
            tmp_alpha = torch.tensor([0.1, 0.05, 0.01,0.15])
            if thresholds is not None:
                tmp_bbox_size = thresholds[pair_idx].repeat(vis.sum()).cpu()
                tmp_threshold = tmp_alpha.unsqueeze(-1) * tmp_bbox_size.unsqueeze(0)
            else:
                tmp_threshold = tmp_alpha * img_size
            if not os.path.exists(f'{save_path}/{category}'):
                os.makedirs(f'{save_path}/{category}')
         
            fig=draw_correspondences_lines(img1_kps[vis][:, [1,0]],res, img2_kps[vis][:, [1,0]], img1, img2, tmp_threshold)
            fig.savefig(f'{save_path}/{category}/{pair_idx}_pred.png')
            fig_gt=draw_correspondences_gathered(img1_kps[vis][:, [1,0]], img2_kps[vis][:, [1,0]], img1, img2)
            fig_gt.savefig(f'{save_path}/{category}/{pair_idx}_gt.png')
            plt.close(fig)
            plt.close(fig_gt)
            current_save_results+=1

        pbar.update(1)

    gt_correspondences = torch.cat(gt_correspondences, dim=0).cpu()
    pred_correspondences = torch.cat(pred_correspondences, dim=0).cpu()
    alpha = torch.tensor([0.1, 0.05, 0.01,0.15,0.2]) if not PASCAL else torch.tensor([0.1, 0.05, 0.15])
    correct = torch.zeros(len(alpha))

    err = (pred_correspondences - gt_correspondences).norm(dim=-1)
    # cal mse
 
    mse = (pred_correspondences - gt_correspondences).pow(2).mean(dim=-1)

    err = err.unsqueeze(0).repeat(len(alpha), 1)
    if thresholds is not None:
        bbox_size = torch.cat(bbox_size, dim=0).cpu()
        threshold = alpha.unsqueeze(-1) * bbox_size.unsqueeze(0)
        correct = err < threshold
  
    print(f'mse: {mse.mean()/len(gt_correspondences)}' )

    correct = correct.sum(dim=-1) / len(gt_correspondences)

    alpha2pck = zip(alpha.tolist(), correct.tolist())
    logger.info(' | '.join([f'PCK-Transfer@{alpha:.2f}: {pck_alpha * 100:.2f}%'
                    for alpha, pck_alpha in alpha2pck]))


    return correct
    
def main(args):
    global TOTAL_SAVE_RESULT, SEED, BBOX_THRE, EDGE_PAD, PASCAL, COUNT_INVIS, SAMPLE, DIST, SD, CAT, VAL, PREVFMAP,CATEGORY, NAME, CROP, MASK, PATH_TO_DINO, PATH_TO_DINO_VAL, PATH_TO_FMAP, PATH_TO_SD, PATH_TO_MASK, PATH_TO_CROP
    TOTAL_SAVE_RESULT = args.TOTAL_SAVE_RESULT
    BBOX_THRE = False if args.IMG_THRESHOLD else True
    EDGE_PAD = args.EDGE_PAD
    SEED = args.SEED
    COUNT_INVIS = True if args.COUNT_INVIS else False
    SAMPLE = args.SAMPLE
    DIST = args.DIST
    MODEL_TYPE = args.MODEL_TYPE
    SD = args.SD
    CAT = args.CAT
    VAL = args.VAL
    PREVFMAP = args.PREVFMAP
    CATEGORY = args.CATEGORY
    NAME = args.NAME
    CROP = args.CROP
    MASK = args.MASK
    
    PATH_TO_DINO = args.PATH_TO_DINO
    PATH_TO_DINO_VAL = args.PATH_TO_DINO_VAL
    PATH_TO_FMAP = args.PATH_TO_FMAP
    PATH_TO_SD = args.PATH_TO_SD
    PATH_TO_MASK = args.PATH_TO_MASK
    PATH_TO_CROP = args.PATH_TO_CROP

    
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.backends.cudnn.benchmark = True
    save_path = f'results/SPair-71k-FMAP-{PREVFMAP}-{FMAP}-VAL-{VAL}-SD-{SD}-CAT-{CAT}' 

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logger.add(f'{save_path}/result.log', format='{time} {level} {message}')

    data_dir = 'data/SPair-71k'
    
    categories = [CATEGORY]
    
    img_size = 840 

    pcks = []
    pcks_05 = []
    pcks_01 = []
    pcks_015 = []
    pcks_020 = []
    start_time=time.time()
    for cat in categories:
        files, kps, thresholds,bboxes = load_spair_data(data_dir, size=img_size, category=cat, subsample=SAMPLE) 
        if BBOX_THRE:
            pck = compute_pck(MODEL_TYPE, save_path, files, kps, cat, dist=DIST, thresholds=thresholds,bbox=bboxes)
        else:
            pck = compute_pck(MODEL_TYPE, save_path, files, kps, cat, dist=DIST)
        pcks.append(pck[0])
        pcks_05.append(pck[1])
        pcks_01.append(pck[2])
        pcks_015.append(pck[3])
        pcks_020.append(pck[4])
        break
    end_time=time.time()
    minutes, seconds = divmod(end_time-start_time, 60)
    
    logger.info(f"Time: {minutes:.0f}m {seconds:.0f}s")
    logger.info(f"Average PCK0.1: {np.average(pcks) * 100:.2f}")
    logger.info(f"Average PCK0.05: {np.average(pcks_05) * 100:.2f}")
    logger.info(f"Average PCK0.01: {np.average(pcks_01) * 100:.2f}") 
    logger.info(f"Average PCK0.15: {np.average(pcks_015) * 100:.2f}") 
    logger.info(f"Average PCK0.2: {np.average(pcks_020) * 100:.2f}")
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', type=int, default=42) 
    parser.add_argument('--SAMPLE', type=int, default=20)                           # sample 20 pairs for each category, set to 0 to use all pairs
    parser.add_argument('--DIST', type=str, default='l2')                           # distance metric, cos, l2, l1, l2_norm, l1_norm, plus, plus_norm
    parser.add_argument('--COUNT_INVIS', action='store_true', default=False)        # set true to count invisible keypoints
    parser.add_argument('--TOTAL_SAVE_RESULT', type=int, default=20)                 # save the qualitative results for the first 5 pairs
    parser.add_argument('--IMG_THRESHOLD', action='store_true', default=False)      # set the pck threshold to the image size rather than the bbox size
    parser.add_argument('--EDGE_PAD', action='store_true', default=False)           # set true to pad the image with the edge pixels               # set true to test on pfpascal dataset
    parser.add_argument('--MODEL_TYPE', type=str, default='dinov2_vitb14', help = 'type of model to extract.')
    parser.add_argument('--SD', type = bool, default= False)
    parser.add_argument('--CAT', type = bool, default= False)
    parser.add_argument('--VAL', type = bool, default = False)
    parser.add_argument('--PREVFMAP', type = bool, default=True)
    parser.add_argument('--CATEGORY',type = str, default='CAT')
    parser.add_argument('--NAME', type = str, default='NAME')
    parser.add_argument('--CROP', type = bool, default=True)
    parser.add_argument('--MASK', type = bool, default=True)
    parser.add_argument('--PATH_TO_FMAP', type = str, default='PATH_TO_FMAP')
    parser.add_argument('--PATH_TO_SD', type = str, default='PATH_TO_SD')
    parser.add_argument('--PATH_TO_DINO', type = str, default='PATH_TO_DINO')
    parser.add_argument('--PATH_TO_DINO_VAL', type = str, default='PATH_TO_DINO_VAL')
    parser.add_argument('--PATH_TO_MASK', type = str, default='PATH_TO_MASK')
    parser.add_argument('--PATH_TO_CROP', type = str, default='PATH_TO_CROP')
    args = parser.parse_args()
    main(args)
    