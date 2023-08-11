import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("C:/Program Files/ASAP 1.8/bin")
import multiresolutionimageinterface as mir

reader = mir.MultiResolutionImageReader()
mr_image = reader.open('E:/CAMELYON16/testing/images731/test_051.tif')
annotation_list = mir.AnnotationList()
xml_repository = mir.XmlRepository(annotation_list)
xml_repository.setSource('E:/CAMELYON16/testing/lesion_annotations/test_051.xml')
xml_repository.load()
annotation_mask = mir.AnnotationToMask()
camelyon17_type_mask = True
label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 1, '_1': 1, '_2': 0}
conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']
output_path = 'E:/CAMELYON16/testing/testmask_051.tif'
annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)

# Get the image data at the highest resolution level
image_data = mr_image.getUCharPatch(0, 0, mr_image.getLevelDimensions(mr_image.getNumberOfLevels() - 1)[0], mr_image.getLevelDimensions(mr_image.getNumberOfLevels() - 1)[1], mr_image.getNumberOfLevels() - 1)

# Display the image
plt.imshow(image_data)
plt.show()
