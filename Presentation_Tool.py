# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:26:24 2017

Module to combine a map image with the original jpg.

Outputs result as a jpg file to preserve quality and also has the option to append a key.

@author: Peter
"""

map_folder = "Images/Maps/"
core_folder = "Images/Cores/"
key_folder = "Images/Other/"
save_folder = "Images/Other/"
key_width = 1527 # pixel width of optional key
spacing = 10 # pixel width of space between images

from PIL import Image

def concat(core_name, add_key=True):
    
    # open core image
    with Image.open(core_folder + core_name + ".jpg") as core_im:
        # get size of core image and use this to define eventual size of combined image
        width, height = core_im.size
        if add_key: full_width = (width * 2) + (spacing * 2) + key_width
        else: full_width = (width * 2) + spacing
        # create template to paste other images into
        combined = Image.new("RGB", (full_width, height), color=(255,255,255))
        # paste the core image into this template
        combined.paste(core_im, (0, 0))
    # unindent to close core image
    # open map image
    with Image.open(map_folder + core_name + ".png") as map_im:
        # paste into the combined image
        combined.paste(map_im, (width + spacing, 0))
    # unindent to close map image
    # open map key
    with Image.open(key_folder + "Large Key" + ".png") as key:
        # paste into the combined image
        combined.paste(key, ((width + spacing)*2, 0))
    # unindent to close map key
    
    # save the new image
    combined.save(save_folder + core_name + "_joined" + ".jpg", format='JPEG', subsampling=0, quality=100)    # save as lossless JPEG
    print("Image saved as " + core_name + "_joined" + ".jpg" + " in " + save_folder)
    
    # display iamge to user then close within python
    combined.show()
    combined.close()

### end Presentation_Tool.concat