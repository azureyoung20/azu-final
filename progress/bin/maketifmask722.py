#!/usr/bin/env python3
import multiresolutionimageinterface as mir
import matplotlib.pyplot as plt
import os
os.add_dll_directory('E:/project rebulit/openslide-win64-20230414/bin')
import openslide
import os.path as osp
from pathlib import Path
import glob


slide_path = 'E:/CAMELYON16/testing/images84/'
anno_path = 'E:/CAMELYON16/testing/lesion_annotations84/'
mask_path = 'D:/tifmask85'
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
    camelyon17_type_mask = False
    label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 255, '_1': 255, '_2': 0}
    conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']
    output_path= osp.join(mask_path, osp.basename(tumor_paths[i]).replace('.tif', '_mask.tif'))
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)
    i=i+1