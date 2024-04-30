import os
import sys
import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from loguru import logger
import argparse
from utils.utils_tss import TSSDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.util_preprocess import resize_
import torch.nn.functional as F 
from os import listdir
from os.path import join


EPS = 1e-10


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


def get_smooth(img, mask=None):

    if mask is not None:
        img_smooth=img.clone().permute(0, 2, 3, 1)
        img_smooth[~mask] = 0
        img=img_smooth.permute(0, 3, 1, 2)

    def _gradient_x(img,mask): #tobe implemented
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def _gradient_y(img,mask):
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy
        
    img_grad_x = _gradient_x(img, mask)
    img_grad_y = _gradient_y(img, mask)

    if mask is not None:
        smooth = (torch.abs(img_grad_x).sum() + torch.abs(img_grad_y).sum())/torch.sum(mask)
    else:
        smooth = torch.mean(torch.abs(img_grad_x)) + torch.mean(torch.abs(img_grad_y))

    return smooth



def nearest_neighbor_flow(src_descriptor, trg_descriptor, ori_shape, mask1=None, mask2=None):
    B, C, H, W = trg_descriptor.shape

    if mask1 is not None and mask2 is not None:
        resized_mask1 = F.interpolate(mask1.cuda().unsqueeze(0).unsqueeze(0).float(), size=src_descriptor.shape[2:], mode='nearest').to(src_descriptor.device)
        resized_mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=trg_descriptor.shape[2:], mode='nearest').to(trg_descriptor.device)
        src_descriptor = src_descriptor * resized_mask1.repeat(1, src_descriptor.shape[1], 1, 1)
        trg_descriptor = trg_descriptor * resized_mask2.repeat(1, trg_descriptor.shape[1], 1, 1)

    real_H, real_W = ori_shape
    long_edge = max(real_H, real_W)
    src_descriptor = src_descriptor.view(B, C, -1).permute(0, 2, 1).squeeze()
    trg_descriptor = trg_descriptor.view(B, C, -1).permute(0, 2, 1).squeeze().to(src_descriptor.device)

    distances = torch.cdist(trg_descriptor, src_descriptor)

    # Find the indices of the minimum distances
    indices = torch.argmin(distances, dim=1)

    indices = indices.view(B, H, W)

    # Convert indices to coordinates
    trg_y = torch.div(indices, W).to(torch.float32)
    trg_x = torch.fmod(indices, W).to(torch.float32)
    # Create coordinate grid
    grid_y, grid_x = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=src_descriptor.device), torch.arange(W, dtype=torch.float32, device=src_descriptor.device))

    # Compare target coordinates with source coordinate grid
    flow_x = trg_x - grid_x
    flow_y = trg_y - grid_y

    # Stack the flow fields together to form the final optical flow
    flow = torch.stack([flow_x, flow_y], dim=1)

    # Perform bilinear interpolation to adjust the optical flow from (60, 60) to (real_H, real_W)

    desc_h, desc_w = flow.shape[2:]
    flow = F.interpolate(flow, size=(real_H,real_W), mode='bilinear', align_corners=False)
    flow *= torch.tensor([real_H / desc_h, real_W / desc_w], dtype=torch.float32, device=src_descriptor.device).view(1, 2, 1, 1)
    # show flow in txt
    flow_ = flow.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
    flow_ = flow_.astype(np.float32)
    np.savetxt('flow.txt', flow_.reshape(-1, 2), fmt='%.3f')
    

    return flow

def load_files_with_dir_name(base_file_name,number,suffix):
    files = [ join(base_file_name,f) for  f in listdir(base_file_name) if f.endswith(suffix)]
    files.sort()
    return files[:number]

def load_files(base_file_name,number,suffix):
    files = []
    for i in range(number):
        files.append(base_file_name+str(i)+suffix)
    return files

def compute_flow(source_img, target_img, batch_num=0, category=['car'], sd_files_1 = None, sd_files_2 = None, dino_files_1 = None, dino_files_2 = None, fmap_files = None):
    if type(category) == str:
        category = [category]
    img_size = 840 
    model_dict={'small':'dinov2_vits14',
                'base':'dinov2_vitb14',
                'large':'dinov2_vitl14',
                'giant':'dinov2_vitg14'}
    
    model_type = model_dict[MODEL_SIZE] 
   
    stride = 14
    device =  'cuda'
    # indiactor = 'v2' if DINOV2 else 'v1'
    # model_size = model_type.split('vit')[-1]
    patch_size = 14
       


    N = 1
    result = []

    for pair_idx in range(N):
        shape = target_img.shape[2:]
        # Load image 1
        img1=Image.fromarray(source_img.squeeze().numpy().transpose(1,2,0).astype(np.uint8))
        img1,width1,height1 = resize_(img1, img_size, resize=True, to_pil=False, edge=False)
        if width1 < height1:
            img1_ = img1[:,(height1-width1)//2:(height1-width1)//2+width1]
        else:
            img1_ = img1[(width1 - height1)//2:(width1 - height1)//2+height1]
        img1 = Image.fromarray(img1_)
        img1 = img1.resize((width1//14*14,height1//14*14))
        width1,height1 = img1.size
        # Load image 2
        
      
        img2 = Image.fromarray(target_img.squeeze().numpy().transpose(1,2,0).astype(np.uint8))

        img2,width2,height2 = resize_(img2, img_size, resize=True, to_pil=False, edge=False)
        if width2 < height2:
            img2_ = img2[:,(height2-width2)//2:(height2-width2)//2+width2]
        else:
            img2_ = img2[(width2 - height2)//2:(width2 - height2)//2+height2]
        img2 = Image.fromarray(img2_)
        img2 = img2.resize((width2//14*14,height2//14*14))
        width2,height2 = img2.size

        with torch.no_grad():
            h1,w1 = height1 // stride, width1 // stride
            h2,w2 = height2 // stride, width2 // stride
            
            img1_desc = torch.load(dino_files_1[batch_num])
            h1,w1 = img1_desc.shape[1:]
            img1_desc = img1_desc.permute(1,2,0)
     
            img1_desc_dino = F.interpolate(img1_desc.permute(2,0,1).unsqueeze(0),size=(h1,w1)).squeeze(0).reshape((img1_desc.shape[-1],-1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2)

            img1_desc = img1_desc.permute(2,0,1)
            img1_desc = img1_desc.reshape((img1_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2) # (1, 1, h*w, dim)
            img2_desc = torch.load(dino_files_2[batch_num])
            h2,w2 = img2_desc.shape[1:]
            img2_desc = img2_desc.permute(1,2,0)
            img2_desc_dino = F.interpolate(img2_desc.permute(2,0,1).unsqueeze(0),size=(h2,w2)).squeeze(0).reshape((img2_desc.shape[-1],-1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2)
        
            img2_desc = img2_desc.permute(2,0,1)
            img2_desc = img2_desc.reshape((img2_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2) # (1, 1, h*w, dim)
                

        
           
            if SD:
                img1_desc_sd = torch.load(sd_files_1[batch_num])

                img1_desc_sd = F.interpolate(img1_desc_sd.unsqueeze(0), size=(h1,w1), mode='bilinear').squeeze(0)
                img1_desc_sd = img1_desc_sd.permute(1,2,0)

                img1_desc_sd = img1_desc_sd.permute(2,0,1)
                img1_desc_sd = img1_desc_sd.reshape((img1_desc_sd.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2) # (1, 1, h*w, dim)
                img2_desc_sd = torch.load(sd_files_2[batch_num])

                img2_desc_sd = F.interpolate(img2_desc_sd.unsqueeze(0), size=(h2,w2), mode='bilinear').squeeze(0)
                img2_desc_sd = img2_desc_sd.permute(1,2,0)

                img2_desc_sd = img2_desc_sd.permute(2,0,1)
                img2_desc_sd = img2_desc_sd.reshape((img2_desc_sd.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2) # (1, 1, h*w, dim)
            
            if SD and not CAT:
                img1_desc = img1_desc_sd
                img2_desc = img2_desc_sd
            

            if SD and CAT:

                img1_desc = img1_desc / (img1_desc.norm(dim=-1, keepdim=True) )
                img2_desc = img2_desc / (img2_desc.norm(dim=-1, keepdim=True)  )
                img1_desc_sd = img1_desc_sd / (img1_desc_sd.norm(dim=-1, keepdim=True) )
                img2_desc_sd = img2_desc_sd / (img2_desc_sd.norm(dim=-1, keepdim=True) )
                img1_desc = img1_desc.to(device)
                img1_desc_sd = img1_desc_sd.to(device)
                img2_desc = img2_desc.to(device)
                img2_desc_sd = img2_desc_sd.to(device)
                img1_desc = torch.cat(( img1_desc_sd,img1_desc), dim=-1)
                img2_desc = torch.cat((img2_desc_sd,img2_desc), dim=-1)
            
            if FMAP and os.path.exists(fmap_files[batch_num]) :
                img2_desc = torch.load(fmap_files[batch_num],map_location=device) 

                img2_desc = img2_desc.permute(2,0,1)
                img2_desc = F.interpolate(img2_desc.unsqueeze(0), size=(h2,w2), mode='bilinear').squeeze(0)
                img1_desc = img1_desc.reshape((h1,w1,-1)).to(device)

                img1_desc = img1_desc.permute(2,0,1)
                img1_desc = img1_desc.reshape((img1_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2) # (1, 1, h*w, dim)
                img2_desc = img2_desc.reshape((img2_desc.shape[0], -1)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2) # (1, 1, h*w, dim)
                
                # raise dimaension
                if SD:
                    img1_desc_dino = img1_desc_dino.to(device)
                    img2_desc_dino = img2_desc_dino.to(device)
                    img1_desc = img1_desc.to(device)
                    img2_desc = img2_desc.to(device)
            
                    img1_desc = img1_desc / (img1_desc.norm(dim=-1, keepdim=True) )
                    img2_desc = img2_desc / (img2_desc.norm(dim=-1, keepdim=True)  )
                    img1_desc_dino = img1_desc_dino / (img1_desc_dino.norm(dim=-1, keepdim=True) )
                    img2_desc_dino = img2_desc_dino / (img2_desc_dino.norm(dim=-1, keepdim=True) )
                    img1_desc = torch.cat((img1_desc, img1_desc_dino), dim = -1)
                    img2_desc = torch.cat((img2_desc, img2_desc_dino), dim = -1)
                
            
            
            img1_desc_reshaped = img1_desc.permute(0,1,3,2).reshape(-1, img1_desc.shape[-1], h1, w1)
            img2_desc_reshaped = img2_desc.permute(0,1,3,2).reshape(-1, img2_desc.shape[-1], h2, w2)

            # compute the flow map based on the nearest neighbor
            result = nearest_neighbor_flow(img1_desc_reshaped, img2_desc_reshaped, shape)

    return result


def run_evaluation_semantic( test_dataloader, device,
                            path_to_save=None, sub_data_name = None):

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_0_05_list, pck_0_01_list, pck_0_1_list, pck_0_15_list = [], [], [], [], [], []
    smooth_est_list, smooth_gt_list = [], []
    eval_buf = {'cls_pck': dict(), 'vpvar': dict(), 'scvar': dict(), 'trncn': dict(), 'occln': dict()}

    # pck curve per image
    pck_thresholds = [0.01]
    pck_thresholds.extend(np.arange(0.05, 0.4, 0.05).tolist())
    pck_per_image_curve = np.zeros((len(pck_thresholds), len(test_dataloader)), np.float32)
    
    sd_prefix = SD_path
    sd_suffix_1 = '_img1.pt'
    sd_files_1 = load_files_with_dir_name(sd_prefix, 200, sd_suffix_1)
    
    sd_suffix_2 = '_img2.pt'
    sd_files_2 = load_files_with_dir_name(sd_prefix, 200, sd_suffix_2)
    
    dino_prefix_1 = DINO_path
    dino_suffix_1 = '_img1.pt'
    dino_files_1 = load_files_with_dir_name(dino_prefix_1, 200, dino_suffix_1)
    
    dino_suffix_2 = '_img2.pt'
    dino_files_2 = load_files_with_dir_name(dino_prefix_1, 200, dino_suffix_2)
    
    fmap_file_prefix = FMAP_path
    
    fmap_file_suffix = f'_tss/TIRAMISU_1999_batch_0_test.pt'
    fmap_files = load_files(fmap_file_prefix, 200, fmap_file_suffix)

    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device)
        mask_valid = mini_batch['correspondence_mask'].to(device)
        category = mini_batch['category']
        
   

        if 'pckthres' in list(mini_batch.keys()):
            L_pck = mini_batch['pckthres'][0].float().item()
        else:
            raise ValueError('No pck threshold in mini_batch')

        flow_est = compute_flow(source_img, target_img, batch_num=i_batch, category=category, sd_files_1=sd_files_1, sd_files_2=sd_files_2, dino_files_1=dino_files_1, dino_files_2=dino_files_2, fmap_files=fmap_files)
       
        smooth_est_list.append(get_smooth(flow_est,mask_valid).cpu().numpy())
        smooth_gt_list.append(get_smooth(flow_gt,mask_valid).cpu().numpy())

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid].to(device)
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid].to(device)


        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()

        epe_all_list.append(epe.view(-1).cpu().numpy())
        mean_epe_list.append(epe.mean().item())
        pck_0_05_list.append(epe.le(0.05*L_pck).float().mean().item())
        pck_0_01_list.append(epe.le(0.01*L_pck).float().mean().item())
        pck_0_1_list.append(epe.le(0.1*L_pck).float().mean().item())
        pck_0_15_list.append(epe.le(0.15*L_pck).float().mean().item())
        for t in range(len(pck_thresholds)):
            pck_per_image_curve[t, i_batch] = epe.le(pck_thresholds[t]*L_pck).float().mean().item()

    epe_all = np.concatenate(epe_all_list)
    pck_0_05_dataset = np.mean(epe_all <= 0.05 * L_pck)
    pck_0_01_dataset = np.mean(epe_all <= 0.01 * L_pck)
    pck_0_1_dataset = np.mean(epe_all <= 0.1 * L_pck)
    pck_0_15_dataset = np.mean(epe_all <= 0.15 * L_pck)
    smooth_est_dataset = np.mean(smooth_est_list)
    smooth_gt_dataset = np.mean(smooth_gt_list)

    output = {'AEPE': np.mean(mean_epe_list), 'PCK_0_05_per_image': np.mean(pck_0_05_list),
              'PCK_0_01_per_image': np.mean(pck_0_01_list), 'PCK_0_1_per_image': np.mean(pck_0_1_list),
              'PCK_0_15_per_image': np.mean(pck_0_15_list),
              'PCK_0_01_per_dataset': pck_0_01_dataset, 'PCK_0_05_per_dataset': pck_0_05_dataset,
              'PCK_0_1_per_dataset': pck_0_1_dataset, 'PCK_0_15_per_dataset': pck_0_15_dataset,
              'pck_threshold_alpha': pck_thresholds, 'pck_curve_per_image': np.mean(pck_per_image_curve, axis=1).tolist()
              }
    logger.info("Validation EPE: %f, alpha=0_1: %f, alpha=0.05: %f" % (output['AEPE'], output['PCK_0_1_per_image'],
                                                                  output['PCK_0_05_per_image']))
    logger.info("smooth_est: %f, smooth_gt: %f" % (smooth_est_dataset, smooth_gt_dataset))

    for name in eval_buf.keys():
        output[name] = {}
        for cls in eval_buf[name]:
            if eval_buf[name] is not None:
                cls_avg = sum(eval_buf[name][cls]) / len(eval_buf[name][cls])
                output[name][cls] = cls_avg

    return output

def main(args):
    global SAMPLE, MODEL_SIZE, SEED, SD, CAT, FMAP, DINO_path, SD_path, FMAP_path

    SAMPLE = args.SAMPLE
    MODEL_SIZE = args.MODEL_SIZE
    SEED = args.SEED
    SD = args.SD
    CAT = args.CAT
    FMAP  = args.FMAP
    DINO_path = args.DINO_path
    SD_path = args.SD_path
    FMAP_path = args.FMAP_path

    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.backends.cudnn.benchmark = True

  #  model, aug = load_model(diffusion_ver=VER, image_size=SIZE, num_timesteps=args.TIMESTEP, block_indices=tuple(INDICES))
    save_path=f'./results_tss/pck_tss_SD_{SD}_CAT_{CAT}_FMAP_{FMAP}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_dir = "data-tss/.s/TSS_CVPR2016"

    class ArrayToTensor(object):
        """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
        def __init__(self, get_float=True):
            self.get_float = get_float

        def __call__(self, array):

            if not isinstance(array, np.ndarray):
                array = np.array(array)
            array = np.transpose(array, (2, 0, 1))
            # handle numpy array
            tensor = torch.from_numpy(array)
            # put it from HWC to CHW format
            if self.get_float:
                # carefull, this is not normalized to [0, 1]
                return tensor.float()
            else:
                return tensor

    co_transform = None
    target_transform = transforms.Compose([ArrayToTensor()])  # only put channel first
    input_transform = transforms.Compose([ArrayToTensor(get_float=False)])  # only put channel first
    output = {}
    for sub_data in [ 'PASCAL']:
        test_set = TSSDataset(os.path.join(data_dir, sub_data),
                                                    source_image_transform=input_transform,
                                                    target_image_transform=input_transform, flow_transform=target_transform,
                                                    co_transform=co_transform,
                                                    num_samples=SAMPLE)
        test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
        results = run_evaluation_semantic( test_dataloader, device='cpu', path_to_save=save_path+'/'+sub_data,sub_data_name = sub_data)
        output[sub_data] = results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument('--SAMPLE', type=int, default=None)
    parser.add_argument('--MODEL_SIZE', type=str, default='base')
    parser.add_argument('--FMAP', action='store_true', default=True)
    parser.add_argument('--SD', action='store_true', default= True)
    parser.add_argument('--CAT', type=bool, default= False)
    parser.add_argument('--DINO_path', type=str, default='data-tss/TSS_CVPR2016/PASCAL_feat_token_11_NOMASK')
    parser.add_argument('--SD_path', type=str, default='data-tss/TSS_CVPR2016/PASCAL_sd_nomask')
    parser.add_argument('--FMAP_path', type=str, default='/home/xinle/fmlib/data-tss/TSS_CVPR2016_/PASCAL-fast-reim-cons-sdbasis-dinoloss/')
    args = parser.parse_args()
    main(args)