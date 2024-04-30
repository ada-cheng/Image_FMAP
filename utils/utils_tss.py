from __future__ import division
import os.path
import numpy as np
import torch.utils.data as data
import cv2
from packaging import version
import torch
import matplotlib.pyplot as plt
from PIL import Image

def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)

    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (h, w, 2))
    return data2D


def pad_to_same_shape(im1, im2, flow, mask):
    # pad to same shape
    if len(im1.shape) == 2:
        im1 = np.dstack([im1,im1,im1])

    if len(im2.shape) == 2:
        im2 = np.dstack([im2,im2,im2])

    if im1.shape[0] <= im2.shape[0]:
        pad_y_1 = im2.shape[0] - im1.shape[0]
        pad_y_2 = 0
    else:
        pad_y_1 = 0
        pad_y_2 = im1.shape[0] - im2.shape[0]
    if im1.shape[1] <= im2.shape[1]:
        pad_x_1 = im2.shape[1] - im1.shape[1]
        pad_x_2 = 0
    else:
        pad_x_1 = 0
        pad_x_2 = im1.shape[1] - im2.shape[1]

    return im1, im2, flow, mask


def make_dataset(dir):
    """For TSS"""
    images = []
    dir_list = [f for f in os.listdir(os.path.join(dir)) if
                os.path.isdir(os.path.join(dir, f))]
    for image_dir in sorted(dir_list):
      
        if image_dir in ['FG3DCar', 'JODS', 'PASCAL']:
            folders_list = [f for f in os.listdir(os.path.join(dir, image_dir)) if
                            os.path.isdir(os.path.join(dir, image_dir, f))]
            for folders in sorted(folders_list):
                img_dir = os.path.join(image_dir, folders)
                cat = None
                if 'Car' in img_dir:
                    cat = 'car'
                else:
                    cat = folders.split('_')[0].lower()
                cat_match_dict={
                    'busd': 'bus',
                    'bike': 'bicycle',
                    'plane': 'aeroplane',
                    'suv': 'car',
                }
                if cat in cat_match_dict.keys():
                    cat = cat_match_dict[cat]
                img1 = os.path.join(img_dir, 'image1.png')
                img2 = os.path.join(img_dir, 'image2.png')
                flow_map = os.path.join(img_dir, 'flow2.flo')
                images.append([[img1, img2], flow_map, cat])


        else:
            if 'Car' in dir:
                cat = 'car'
            else:
                cat = image_dir.split('_')[0].lower()
            if cat in ['busd', 'bike', 'plane', 'suv']:
                cat_match_dict = {
                    'busd': 'bus',
                    'bike': 'bicycle',
                    'plane': 'aeroplane',
                    'suv': 'car',
                }
                cat = cat_match_dict[cat]
            img_dir = image_dir
            # the flow is taken both ways
            img1 = os.path.join(img_dir, 'image1.png') 
            img2 = os.path.join(img_dir, 'image2.png')  
            flow_map = os.path.join(img_dir, 'flow2.flo')
            images.append([[img1, img2], flow_map, cat])


    return images


def flow_loader(root, path_imgs, path_flo):
    imgs = [os.path.join(root, path) for path in path_imgs]

    flo = os.path.join(root, path_flo)
    flow = load_flo(flo)
    base_path = os.path.dirname(path_flo)
    image_number = path_flo[-5] # getting the mask number, either 1 or 2 depending which image is the target !
    path_mask = os.path.join(root, base_path, 'mask{}.png'.format(image_number))
    mask = cv2.imread(path_mask, 0)/255 # before it was 255, we want mask in range 0,1
    images = [  np.asarray(Image.open(img).convert('RGB')) for img in imgs]
    source_size = images[0].shape # threshold is max size of source image for pck
    im1, im2, flow, mask = pad_to_same_shape(images[0], images[1], flow, mask)
    return [im1, im2], flow, mask.astype(np.uint8), source_size


def flow_loader_with_paths(root, path_imgs, path_flo):
    imgs = [os.path.join(root, path) for path in path_imgs]

    flo = os.path.join(root, path_flo)
    flow = load_flo(flo)
    base_path = os.path.dirname(path_flo)
    image_number = path_flo[-5]  # getting the mask number, either 1 or 2 depending which image is the target !
    path_mask = os.path.join(root, base_path, 'mask{}.png'.format(image_number))
    mask = cv2.imread(path_mask, 0)/255  # before it was 255, we want mask in range 0,1
    images =  [  np.asarray(Image.open(img).convert('RGB')) for img in imgs]
    source_size = images[0].shape # threshold is max size of source image for pck
    target_size = images[1].shape
    im1, im2, flow, mask = pad_to_same_shape(images[0], images[1], flow, mask)

    return [im1, im2], flow, mask.astype(np.uint8), source_size, target_size, path_flo


class TSSDataset(data.Dataset):
    """TSS dataset. Builds the dataset of TSS image pairs and corresponding ground-truth flow fields."""
    def __init__(self, root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                 co_transform=None, num_samples=None):
        """
        Args:
            root: path to root folder
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            co_transform: transformations to apply to both images and ground-truth flow fields
            split: split (float) between training and testing, 0 means all pairs are in test_dataset
        Output in __getittem__:
            source_image
            target_image
            flow_map
            correspondence_mask: valid correspondences (only on foreground objects here)
            source_image_size
            target_image_size
        """
        test_list = make_dataset(root)
        self.root = root
        if num_samples is not None:
            test_list = test_list[:num_samples]

       
        self.path_list = test_list
        self.first_image_transform = source_image_transform
        self.second_image_transform = target_image_transform
        self.target_transform = flow_transform
        self.co_transform = co_transform
        self.loader = flow_loader

    def __getitem__(self, index):
        """
        Args:
            index:

        Returns: Dictionary with fieldnames:
            source_image
            target_image
            flow_map
            correspondence_mask: valid correspondences (only on foreground objects here)
            source_image_size
            target_image_size
        """
        inputs, target, cat = self.path_list[index ]
        source_img_path = inputs[0]
        target_img_path = inputs[1]
        inputs, target, mask, source_size, target_size, path_flo = flow_loader_with_paths(self.root, inputs, target)
        
        if self.first_image_transform is not None:
            inputs[0] = self.first_image_transform(inputs[0])
        if self.second_image_transform is not None:
            inputs[1] = self.second_image_transform(inputs[1])
        if self.target_transform is not None:
            target = self.target_transform(target)
        L_pck = float(max((target_size[0],target_size[1], source_size[0],source_size[1])))
        return {'source_image': inputs[0],
                'target_image': inputs[1],
                'flow_map': target,
                'correspondence_mask': mask.astype(np.bool_) if version.parse(torch.__version__) >= version.parse("1.1")
                    else mask.astype(np.uint8),
                'source_image_size': np.array(source_size),
                'target_image_size': np.array(target_size),
                'pckthres': L_pck,
                'category': cat,
                'source_img_path': source_img_path,
                'target_img_path': target_img_path,
                }

    def __len__(self):
        return len(self.path_list)