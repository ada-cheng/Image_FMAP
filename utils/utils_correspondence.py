import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple
import faiss
import cv2
import os
from matplotlib.patches import ConnectionPatch

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
    return canvas

def co_pca(features1, features2, dim=[128,128,128]):
    
    processed_features1 = {}
    processed_features2 = {}
    
    s5_size_h_1 = features1['s5'].shape[-2]
    s5_size_w_1 = features1['s5'].shape[-1]
    num_patch_s5_1 = s5_size_h_1 * s5_size_w_1
    
    s4_size_h_1 = features1['s4'].shape[-2]
    s4_size_w_1 = features1['s4'].shape[-1]
    num_patch_s4_1 = s4_size_h_1 * s4_size_w_1
    
    s3_size_h_1 = features1['s3'].shape[-2]
    s3_size_w_1 = features1['s3'].shape[-1]
    num_patch_s3_1 = s3_size_h_1 * s3_size_w_1
    
    s5_size_h_2 = features2['s5'].shape[-2]
    s5_size_w_2 = features2['s5'].shape[-1]
    num_patch_s5_2 = s5_size_h_2 * s5_size_w_2
    
    s4_size_h_2 = features2['s4'].shape[-2]
    s4_size_w_2 = features2['s4'].shape[-1]
    num_patch_s4_2 = s4_size_h_2 * s4_size_w_2
    
    s3_size_h_2 = features2['s3'].shape[-2]
    s3_size_w_2 = features2['s3'].shape[-1]
    num_patch_s3_2 = s3_size_h_2 * s3_size_w_2
    
    
    # Get the feature tensors
    s5_1 = features1['s5'].reshape(features1['s5'].shape[0], features1['s5'].shape[1], -1)
    s4_1 = features1['s4'].reshape(features1['s4'].shape[0], features1['s4'].shape[1], -1)
    s3_1 = features1['s3'].reshape(features1['s3'].shape[0], features1['s3'].shape[1], -1)

    s5_2 = features2['s5'].reshape(features2['s5'].shape[0], features2['s5'].shape[1], -1)
    s4_2 = features2['s4'].reshape(features2['s4'].shape[0], features2['s4'].shape[1], -1)
    s3_2 = features2['s3'].reshape(features2['s3'].shape[0], features2['s3'].shape[1], -1)
    # Define the target dimensions
    target_dims = {'s5': dim[0], 's4': dim[1], 's3': dim[2]}

    # Compute the PCA
    for name, tensors,patch_num in zip(['s5', 's4', 's3'], [[s5_1, s5_2], [s4_1, s4_2], [s3_1, s3_2]],[num_patch_s5_1, num_patch_s4_1, num_patch_s3_1]):
        target_dim = target_dims[name]

        # Concatenate the features
        features = torch.cat(tensors, dim=-1) # along the spatial dimension
        features = features.permute(0, 2, 1) # Bx(t_x+t_y)x(d)

        # Compute the PCA
        # pca = faiss.PCAMatrix(features.shape[-1], target_dim)

        # Train the PCA
        # pca.train(features[0].cpu().numpy())

        # Apply the PCA
        # features = pca.apply(features[0].cpu().numpy()) # (t_x+t_y)x(d)

        # convert to tensor
        # features = torch.tensor(features, device=features1['s5'].device).unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        
        
        # equivalent to the above, pytorch implementation
        mean = torch.mean(features[0], dim=0, keepdim=True)
        centered_features = features[0] - mean
        U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
        reduced_features = torch.matmul(centered_features, V[:, :target_dim]) # (t_x+t_y)x(d)
        features = reduced_features.unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        

        # Split the features
        processed_features1[name] = features[:, :, :patch_num]
        processed_features2[name] = features[:, :, patch_num:]

    # reshape the features
    
    processed_features1['s5']=processed_features1['s5'].reshape(processed_features1['s5'].shape[0], -1, s5_size_h_1, s5_size_w_1)
    processed_features1['s4']=processed_features1['s4'].reshape(processed_features1['s4'].shape[0], -1, s4_size_h_1, s4_size_w_1)
    processed_features1['s3']=processed_features1['s3'].reshape(processed_features1['s3'].shape[0], -1, s3_size_h_1, s3_size_w_1)

    processed_features2['s5']=processed_features2['s5'].reshape(processed_features2['s5'].shape[0], -1, s5_size_h_2, s5_size_w_2)
    processed_features2['s4']=processed_features2['s4'].reshape(processed_features2['s4'].shape[0], -1, s4_size_h_2, s4_size_w_2)
    processed_features2['s3']=processed_features2['s3'].reshape(processed_features2['s3'].shape[0], -1, s3_size_h_2, s3_size_w_2)

    # Upsample s5 spatially by a factor of 2
    processed_features1['s5'] = F.interpolate(processed_features1['s5'], size=(processed_features1['s4'].shape[-2:]), mode='bilinear', align_corners=False)
    processed_features2['s5'] = F.interpolate(processed_features2['s5'], size=(processed_features2['s4'].shape[-2:]), mode='bilinear', align_corners=False)

    # Concatenate upsampled_s5 and s4 to create a new s5
    processed_features1['s5'] = torch.cat([processed_features1['s4'], processed_features1['s5']], dim=1)
    processed_features2['s5'] = torch.cat([processed_features2['s4'], processed_features2['s5']], dim=1)

    # Set s3 as the new s4
    processed_features1['s4'] = processed_features1['s3']
    processed_features2['s4'] = processed_features2['s3']

    # Remove s3 from the features dictionary
    processed_features1.pop('s3')
    processed_features2.pop('s3')

    # current order are layer 8, 5, 2
    features1_gether_s4_s5 = torch.cat([processed_features1['s4'], F.interpolate(processed_features1['s5'], size=(processed_features1['s4'].shape[-2:]), mode='bilinear')], dim=1)
    features2_gether_s4_s5 = torch.cat([processed_features2['s4'], F.interpolate(processed_features2['s5'], size=(processed_features2['s4'].shape[-2:]), mode='bilinear')], dim=1)

    return features1_gether_s4_s5, features2_gether_s4_s5


def generate_cute_rainbow(mask1, mask2, image1, image2, features1, features2, height1,width1,height2,width2):
    device_id = 0
    def polar_color_map(image_shape):
        h, w = image_shape[:2]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)

        # Find the center of the mask
        mask=mask2.cpu()
        mask_center = np.array(np.where(mask > 0))
        mask_center = np.round(np.mean(mask_center, axis=1)).astype(int)
        mask_center_y, mask_center_x = mask_center

        # Calculate distance and angle based on mask_center
        xx_shifted, yy_shifted = xx - x[mask_center_x], yy - y[mask_center_y]
        max_radius = np.sqrt(h**2 + w**2) / 2
        radius = np.sqrt(xx_shifted**2 + yy_shifted**2) * max_radius
        angle = np.arctan2(yy_shifted, xx_shifted) / (2 * np.pi) + 0.5

        angle = 0.2 + angle * 0.6  # Map angle to the range [0.25, 0.75]
        radius = np.where(radius <= max_radius, radius, max_radius)  # Limit radius values to the unit circle
        radius = 0.2 + radius * 0.6 / max_radius  # Map radius to the range [0.1, 1]

        return angle, radius
    
    
    if height1 is not None and width1 is not None: # resize the feature map to the resolution
        features1 = F.interpolate(features1.unsqueeze(0), size=(height1, width1), mode='bilinear')
        features2 = F.interpolate(features2.unsqueeze(0), size=(height2, width2), mode='bilinear')
    # resize the image to the shape of the feature map
    if isinstance(image1, torch.Tensor):
        resized_image1 = F.interpolate(image1.permute(2, 0, 1).unsqueeze(0), size=(height1, width1), mode='bilinear').squeeze(0).permute(1, 2, 0)
        resized_image2 = F.interpolate(image2.permute(2, 0, 1).unsqueeze(0), size=(height2, width2), mode='bilinear').squeeze(0).permute(1, 2, 0)
    else:
        resized_image1 = image1.resize((width1, height1))
        resized_image2 = image2.resize((width2, height2))

    # change image to numpy array
    resized_image1 = np.array(resized_image1)
    resized_image2 = np.array(resized_image2)
    resized_image1 = torch.tensor(resized_image1).to(f"cuda:{device_id}").float()
    resized_image2 = torch.tensor(resized_image2).to(f"cuda:{device_id}").float()

    mask1 = F.interpolate(mask1.cuda(device_id).unsqueeze(0).unsqueeze(0).float(), size=resized_image1.shape[:2], mode='nearest').squeeze(0).squeeze(0)
    mask2 = F.interpolate(mask2.cuda(device_id).unsqueeze(0).unsqueeze(0).float(), size=resized_image2.shape[:2], mode='nearest').squeeze(0).squeeze(0)

    # Mask the images
    resized_image1 = resized_image1 * mask1.unsqueeze(-1).repeat(1, 1, 3)
    resized_image2 = resized_image2 * mask2.unsqueeze(-1).repeat(1, 1, 3)
    # Normalize the images to the range [0, 1]
    resized_image1 = (resized_image1 - resized_image1.min()) / (resized_image1.max() - resized_image1.min())
    resized_image2 = (resized_image2 - resized_image2.min()) / (resized_image2.max() - resized_image2.min())

    angle, radius = polar_color_map(resized_image2.shape)

    angle_mask = angle * mask2.cpu().numpy()
    radius_mask = radius * mask2.cpu().numpy()

    hsv_mask = np.zeros(resized_image2.shape, dtype=np.float32)
    hsv_mask[:, :, 0] = angle_mask
    hsv_mask[:, :, 1] = radius_mask
    hsv_mask[:, :, 2] = 1
 

    rainbow_mask2 = cv2.cvtColor((hsv_mask * 255).astype(np.uint8), cv2.COLOR_HSV2BGR) / 255

    # Apply the rainbow mask to image2
    mask2[(mask2>0)] = 1
    rainbow_image2 = rainbow_mask2 * mask2.cpu().numpy()[:, :, None]

    # Create a white background image
    background_color = np.array([1, 1, 1], dtype=np.float32)
    background_image = np.ones(resized_image2.shape, dtype=np.float32) * background_color

    # Apply the rainbow mask to image2 only in the regions where mask2 is 1
    rainbow_image2 = np.where(mask2.cpu().numpy()[:, :, None] == 1, rainbow_mask2, background_image)
    
    
    return rainbow_image2