import cv2
import torch
from tqdm import tqdm
import numpy as np
import os
from typing import List, Tuple
from loguru import logger
import argparse
import time
from glob import glob
import json
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt


def resize_(img, target_res, resize=True, to_pil=True, edge=False):
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
            canvas[(width - height) // 2: (width + height) // 2,:] = img
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
    return canvas,width, height

def foreground_preprocess(annotation_path, image_path):
    """
    return a PIL image, an array, and an array
    """
    # Load the annotation mask (border of the foreground object)
    annotation_mask = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
    foreground_mask = annotation_mask

    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colored_image = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
    fill_color = 255
    cv2.drawContours(colored_image, contours, -1, fill_color, thickness=cv2.FILLED)

    # Load the original image
    image = cv2.imread(image_path)

    # Apply the foreground mask using a bitwise AND operation
    foreground = cv2.bitwise_and(image, image, mask=colored_image)
    
    foreground_pil = Image.fromarray(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))

    return foreground_pil,colored_image,image