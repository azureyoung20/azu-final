import sys
sys.path.append("C:/Program Files/ASAP 2.1/bin")
import multiresolutionimageinterface as mir
import matplotlib.pyplot as plt
import cv2
import numpy as np
import xml.etree.ElementTree as et
import pandas as pd

image_reader = mir.MultiResolutionImageReader()
image_path = 'E:/CAMELYON16/testing/images731/test_051.tif'
image_handle = image_reader.open(image_path)
image_width, image_height = image_handle.getDimensions()
level_dimensions = image_handle.getLevelDimensions(4)

def convert_xml_to_df(xml_file):
    parseXML = et.parse(xml_file)
    root = parseXML.getroot()
    df_cols = ['Object', 'X', 'Y']
    df_xml = pd.DataFrame(columns=df_cols)
    for annotation in root.iter('Annotation'):
        obj_name = annotation.attrib.get('Name')
        for coordinate in annotation.iter('Coordinate'):
            x_coord = float(coordinate.attrib.get('X'))
            x_coord = ((x_coord) * level_dimensions[0]) / image_width
            y_coord = float(coordinate.attrib.get('Y'))
            y_coord = ((y_coord) * level_dimensions[1]) / image_height
            df_xml = df_xml._append(pd.Series([obj_name, x_coord, y_coord], index=df_cols), ignore_index=True)
    return df_xml

annotations_df = convert_xml_to_df('E:/CAMELYON16/testing/lesion_annotations/test_051.xml')

def remove_duplicates(duplicate_list):
    final_list = []
    for item in duplicate_list:
        if item not in final_list:
            final_list.append(item)
    return final_list

unique_objects = remove_duplicates(annotations_df['Object'])

coxy_list = [[] for x in range(len(unique_objects))]

i = 0
for obj in unique_objects:
    new_x_coords = annotations_df[annotations_df['Object'] == obj]['X']
    new_y_coords = annotations_df[annotations_df['Object'] == obj]['Y']
    print(obj)
    print(new_x_coords, new_y_coords)
    new_xy_coords = list(zip(new_x_coords, new_y_coords))
    coxy_list[i] = np.array(new_xy_coords, dtype=np.int32)
    i += 1

image_tile = image_handle.getUCharPatch(0, 0, level_dimensions[0], level_dimensions[1], 4)

cv2.drawContours(image_tile, coxy_list, -1, (0, 255, 0), 10)

cv2.imwrite("D:/Annotated tif/annotated_image51.jpg", image_tile)
