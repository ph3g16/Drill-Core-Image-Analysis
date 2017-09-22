# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:30:53 2017

Extends an image by appending some of the left columns to the right.

This is to allow image analysis on the full 360 degrees of rock core (analysis can be impresise at border areas).

The extended image is displayed to confirm that the right hand side has a true and smooth transition at 360'+.
If the cut can be discerned visually then extended images should not be used for anlysis.

@author: Peter
"""

####### !!!!!! sample images already have some overlap. Estimate 14 pixel overlap (from 2A_114_2_1) - this varies between samples.

from PIL import Image
import math
import os

def extend(file_name, copy_width, overwrite=0, save=False):
# copy_width should be calculated according to intended sampling method (step_size and segment_dimension)
# overwrite specifies the number of columns to lose on the right hand side (my data had a single pixel black band on the right side of every scan)
    
    target_folder = "Images/Cores/"
    destination_folder = "Images/Rolled/"
    
    im = Image.open(target_folder + file_name + ".jpg")
    
    width, height = im.size
    box = (0, 0, copy_width, height)
    column = im.crop(box)
    
    new_im = Image.new("RGB", (width + copy_width - overwrite, height))
    
    new_im.paste(im, (0, 0))
    new_im.paste(column, (width - overwrite, 0))
    
    if save:
        new_im.save(destination_folder + file_name + ".png")
    else: new_im.show()
    
### end Cylindrical_Image_Prep.extend
    
def trim(file_name, save=False, receive_mode=".jpg", delete_old=False):
# trim width should be based on the sampling method (step_size and segment_dimension)
# intention is to extend images first (for analysis) then trim them back to a size which corresponds with the map output

    target_folder = "Images/Cores/"
    destination_folder = "Images/Cores/"
    
    im = Image.open(target_folder + file_name + receive_mode)
    
    width, height = im.size
    segment_dimension = 12
    step_size = 6
    new_x = math.floor((width - segment_dimension) / step_size) + 1
    new_y = math.floor((height - segment_dimension) / step_size) + 1
    
    print(new_x)
    print(new_y)
    
    # calculate border dimensions
    top_and_left_trim = int((segment_dimension - step_size) / 2)
    right_trim = int(top_and_left_trim + (width - (top_and_left_trim * 2 + step_size * new_x)))
    lower_trim = int(top_and_left_trim + (height - (top_and_left_trim * 2 + step_size * new_y)))

    box = (top_and_left_trim, top_and_left_trim, width - right_trim, height - lower_trim)
    cropped_im = im.crop(box)

    if save:
        cropped_im.save(destination_folder + file_name + ".png")
        if delete_old:
            os.remove(target_folder + file_name + receive_mode)
    else: cropped_im.show()
    

    
### end Cylindrical_Image_Prep.trim