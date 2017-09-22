# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 12:46:32 2017

Checks an image to see if any bright green is present.

Use this to validate the use of (0,255,1) as alpha channel in training images.

Predicated on the use of a JPEG dataset.

@author: Peter
"""

from PIL import Image

def validate(file_name):
    
    target_directory = "Images/Cores/"
    alpha_channel = (0,255,1)
    
    im = Image.open(target_directory + file_name + ".jpg")
    print(im.format,im.size,im.mode)
    
    i_length, j_length = im.size
    count = 0
    
    for j in range(j_length):
        for i in range(i_length):
            if im.getpixel((i,j)) == alpha_channel:
                count += 1
    
    print("{} ugly green pixels present".format(count))
    im.close() 
    return(count)     
    
### end Alpha_Validation.validate