#!/usr/bin/env python3
import multiresolutionimageinterface as mir
import matplotlib.pyplot as plt
import os
import numpy as np
os.add_dll_directory('E:/project rebulit/openslide-win64-20230414/bin')
import openslide
import os.path as osp
from pathlib import Path
import glob
from PIL import Image


slide_path = 'E:/CAMELYON16/testing/images84/'
anno_path = 'E:/CAMELYON16/testing/lesion_annotations84/'
mask_path = 'D:/tifmask84'
tumor_paths = glob.glob(osp.join(slide_path, '*.tif'))
tumor_paths.sort()
anno_tumor_paths = glob.glob(osp.join(anno_path, '*.xml'))
anno_tumor_paths.sort()

reader = mir.MultiResolutionImageReader()
i=0
while i < len(tumor_paths):
    mr_image = reader.open(tumor_paths[i])
    annotation_list=mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(anno_tumor_paths[i])
    xml_repository.load()
    annotation_mask=mir.AnnotationToMask()
    label_map = {'metastases': 1, 'normal': 2}
    conversion_order = ['metastases', 'normal']
    output_path= osp.join(mask_path, osp.basename(tumor_paths[i]).replace('.tif', '_mask.tif'))
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)

    # Open the mask with openslide
    mask = openslide.OpenSlide(output_path)

    # Calculate the new dimensions for level 5
    new_dims = tuple(np.array(mask.level_dimensions[0]) // 32)  # 32 = 2^5

    # Read the mask at level 5 and save it
    mask_data = mask.read_region((0,0), 5, new_dims)
    mask_data.save(output_path)

    i=i+1
