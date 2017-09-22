# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:23:55 2017

Create a "map" file based on human input rather than classification software input.

This is useful for end validation. Allows comparison between human and machine classifications.

Also used to create maps identifying areas which are inappropriate for classifications.
e.g. out of focus areas or bits of carboard.
These can easily be identified by hand and should be removed prior to rigorous analysis.
It is easier to remove these features after classification (rather than before) as otherwise would have to write software constructing non-rectangular maps or would have to train classifier to recognise at least one additional category.

Operation instructions:
    - open original image in GIMP (or equivalent)
    - select required area. Two methods:
        * scissors select tool, select area then fill with appropriae filter colour.
        * draw around area using 1-pixel pencil, select pencil lines, copy to new image, fill within the lines, transplant back to original image (to ensure correct orientation).
    - save/export image as .png
    - use this module to turn the .png into a map

@author: Peter
"""

import numpy
from PIL import Image

def map_zones(file_name):
    
    target_folder = "Images/Validation/"
    destination_folder = "Binaries/Validation/"
    
    cat0 = (0, 255, 0) # green, rock
    cat1 = (255, 0, 0) # red, laumonite
    cat2 = (128, 128, 128) # grey, pren
    cat3 = (0, 0, 0) # black, fault
    cat4 = (255, 0, 255) # magenta, blue crayon 
    cat5 = (255, 255, 0) # yellow, yellow crayon
    cat6 = (0, 0, 255) # blue, altered gabbro
    cat7 = (128, 128, 0) # olive, olivines
    cat8 = (255, 192, 0) # orange, ? clinozoisite
    cat9 = (0, 255, 255) # cyan, red crayon  
    
    # open the human generated PNG
    im = Image.open(target_folder + file_name + ".png")
    
    # get image dimension
    x, y = im.size
    
    # iterate through the image to generate a map file corresponding to zone input
    stack = []
    for j in range(y):
        row = []
        for i in range(x):
            if im.getpixel((i,j)) == cat0:
                row.append(0)
            elif im.getpixel((i,j)) == cat1:
                row.append(1)
            elif im.getpixel((i,j)) == cat2:
                row.append(2)
            elif im.getpixel((i,j)) == cat3:
                row.append(3)
            elif im.getpixel((i,j)) == cat4:
                row.append(4)
            elif im.getpixel((i,j)) == cat5:
                row.append(5)
            elif im.getpixel((i,j)) == cat6:
                row.append(6)
            elif im.getpixel((i,j)) == cat7:
                row.append(7)
            elif im.getpixel((i,j)) == cat8:
                row.append(8)
            elif im.getpixel((i,j)) == cat9:
                row.append(9)
            else:
                print("map contains erroneous pixel: {} at position {},{}".format(im.getpixel((i, j)), i, j))
                return
        stack.append(row)
    
    # we have our final output
    # repackage this as a numpy array and save for later use             
    output = numpy.asarray(stack,numpy.uint8)
    numpy.save(destination_folder + file_name + ".npy", output)
    
    print("Map saved as {}.npy".format(destination_folder + file_name))
    
### end Human_Map.map_zones

def map_bounds(file_name):
    
    target_folder = "Images/Bounds/"
    destination_folder = "Binaries/Boundary_Maps/"
    
    alpha_channel = (0, 255, 0)
    step_size = 3
    
    
    # open the human generated PNG
    im = Image.open(target_folder + file_name + ".png")
    
    # get image dimension
    x, y = im.size
    
    # iterate through the image to generate a map file corresponding to filtered area
    stack = []
    for j in range(y):
        row = []
        for i in range(x):
            if im.getpixel((i,j)) == alpha_channel:
                # mark pixel as bad/filtered
                row.append(255)
            else:
                # mark pixel as useful
                row.append(0)
        stack.append(row)
    
    # reduce map to correct resolution    
    stack = bin_data(stack, step_size)
    
    # we have our final output
    # repackage this as a numpy array and save for later use             
    output = numpy.asarray(stack,numpy.uint8)
    numpy.save(destination_folder + file_name + "_bounds" + ".npy", output)
    
### end Human_Map.map_bounds

def bin_data(stack, loss_ratio):
# reduces the map resolution by binning a huge amount of data (if loss_ratio=2 it bins 3/4 of the data)
# in this case we want to bin data equivalent to step_size to bring the map to a resolution which matches the machine classifications
 
    stack = stack[::loss_ratio]
    
    lossy_stack = []
    for element in stack:
        lossy_stack.append(element[::loss_ratio])
    
    return(lossy_stack)

### end Human_Map.bin_data