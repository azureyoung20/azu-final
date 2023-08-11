import os
import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from PIL import Image

def extract_tissue_mask(wsi_path, level=6, RGB_min=50):
    slide = openslide.OpenSlide(wsi_path)

    img_RGB = np.transpose(np.array(slide.read_region((0, 0),
                           level,
                           slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])

    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

    return tissue_mask

def save_patches(tissue_mask, output_dir, patch_size=768):
    for i in range(0, tissue_mask.shape[0], patch_size):
        for j in range(0, tissue_mask.shape[1], patch_size):
            patch = tissue_mask[i:i+patch_size, j:j+patch_size]
            Image.fromarray((patch * 255).astype(np.uint8)).save(f"{output_dir}/patch_{i}_{j}.png")

# Usage:
image_path = "E:/CAMELYON16/testing/images/test_001.tif"
output_dir = "F:/generated test patches"
tissue_mask = extract_tissue_mask(image_path)
save_patches(tissue_mask, output_dir)


