import cv2
import torch
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch

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

    def compute_correct():
        alpha = torch.tensor([0.1, 0.05, 0.01])
        correct = torch.zeros(len(alpha))
        err = (torch.tensor(points2) - torch.tensor(gt_points2)).norm(dim=-1)
        err = err.unsqueeze(0).repeat(len(alpha), 1)
        correct = err < threshold.unsqueeze(-1) if len(threshold.shape)==1 else err < threshold
        return correct

    correct = compute_correct()[0]
    # print(correct.shape, len(points1)) 

    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)

    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                            "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 0.03*max(image1.size), 0.01*max(image1.size)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)
    ax1.set_xlim(0, image1.size[0])
    ax1.set_ylim(image1.size[1], 0)
    ax2.set_xlim(0, image2.size[0])
    ax2.set_ylim(image2.size[1], 0)

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

def resize_feat(feat, target_res, resize=True, to_pil=True, edge=False):
    """
    feat: pytorch tensor (h,w,dim)
    """
    original_height, original_width = feat.shape[0], feat.shape[1]
    original_channels = feat.shape[2] 
    if not edge:
        if original_channels == 1:
            canvas = torch.zeros([target_res, target_res,1], dtype=torch.float32)
        else:
            canvas = torch.zeros([target_res, target_res, original_channels], dtype=torch.float32)
        if original_height <= original_width:
            if resize:
                feat = F.interpolate(feat.unsqueeze(0).permute(0,3,1,2), size=( int(np.around(target_res * original_height / original_width)),target_res), mode='bilinear', align_corners=True)
                feat = feat.squeeze(0).permute(1,2,0)
            height,width = feat.shape[0], feat.shape[1]
            canvas[abs(height - width) // 2: (width + height) // 2,:] = feat
            
        else:
            if resize:
                feat = F.interpolate(feat.unsqueeze(0).permute(0,3,1,2), size=(target_res,int(np.around(target_res * original_width / original_height))), mode='bilinear', align_corners=True)
                feat = feat.squeeze(0).permute(1,2,0)
            width, height = feat.shape[0], feat.shape[1]
            canvas[:,abs(width - height) // 2: (height + width) // 2] = feat
    return canvas
   