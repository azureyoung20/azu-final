import os
import openslide
from PIL import Image


def Tif_To_Png(TIF_img_dir, PNG_img_dir):
    if not os.path.exists(PNG_img_dir):
        os.mkdir(PNG_img_dir)

    TIF_names = os.listdir(TIF_img_dir)
    for name in TIF_names:
        absolute_path = os.path.join(TIF_img_dir, name)
        slide = openslide.OpenSlide(absolute_path)

        # Read a small thumbnail image (downsampled)
        thumbnail = slide.get_thumbnail((slide.dimensions[0] // 16, slide.dimensions[1] // 16))

        # Save the thumbnail as a PNG
        png_name = os.path.join(PNG_img_dir, name.rsplit('.', 1)[0] + '.png')
        thumbnail.save(png_name)


if __name__ == '__main__':
    TIF_dir = 'F:/tif to png/tif'
    PNG_dir = 'F:/tif to png/png'

    Tif_To_Png(TIF_img_dir=TIF_dir, PNG_img_dir=PNG_dir)

