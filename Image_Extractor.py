# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:54:11 2017

Written for Python 3.5 with Pillow extension.

@author: Peter
"""

# code opens a JPG file and extracts a series of new images thus segmenting the parent image into an easily digestible format (for machine learning)

# aim is to generate a series of images from a larger texture which has been pre-classified

# accepts any input size larger than the sampling window

# following can/should be adjusted according to task:
    # sample size
    # step size
    # background filter
    
# non-rectangular parent images should be transplanted into an image with a background of the appropriate filter colour
# program explicitely avoids sampling any segments containing background pixels (hence filter should be selected from colours not used in the image itself)

import numpy

def harvest(image_name, category, alpha_channel, segment_dimension, step_size, limit, portion=False, HSV=False):
    # image_name should be the parent file name string
    # category variable will tell the program what label to attach
    # alpha_channel can be set to false to disable filtering
        # alternatively it could be a RGB vector specifing the colour to treat as background
    
    from PIL import Image    #homework
    
    im = Image.open(image_name)
    
    if HSV: im = im.convert("HSV")
    
    if portion is not False:    # "is not" is a single operator and could be restated "not portion"
        box = (portion[0], portion[1], portion[2], portion[3])
        im = im.crop(box)
    else:
        print(im.format,im.size,im.mode)        # prints image properties, for reassurance/debugging
    
    i_length, j_length = im.size        # finds image size and defines this for both axes
    
    countback = False               # initialise variable - countback is used to tell the extractor to go back if it found a valid image but the previous image was invalid (due to filter)
                                    # this makes it easier for the program to pick up data from thin crops but biases the system towards collecting data from left side of image
    
    big_list = []
    cycle = 0           # initialises a count for the number of images collected - used for labelling images
                        # i and j are not used to label the images as these may be inconsistent due to images skipped by filter colour
                        # using a seperate count variable also makes it easy to build a dataset from multiple parent images
                                            
    
    #iterate over i (x-axis) and j (y-axis) to find all valid segments    
    for j in range(0, j_length - segment_dimension + 1, step_size):
        if cycle >= limit: break
            
        for i in range(0, i_length - segment_dimension + 1, step_size):
            if cycle >= limit: break
            
            box = (i, j, i+segment_dimension, j+segment_dimension)
            segment = im.crop(box)                                      # creates segment as an image object of appropriate dimension
            if alpha_channel is not False:      # "not False" is used because alpha_channel is never True. Will be false or a RGB input.
                if check_filter(segment_dimension, segment, alpha_channel) is True:         # are there any background/filter pixels in the segment?
                    countback = True                                        # - if yes then skip this segment and go to next cycle (i.e. we want to filter out any segments containing the filter pixel)
                    continue                                                # - if no then you found a valid segment, add this data to the export list
                else: 
                    if countback is True:
                        for stepback in range(0, step_size, 1):
                            i = i - 1
                            box = (i, j, i+segment_dimension, j+segment_dimension)
                            segment = im.crop(box)
                            if check_filter(segment_dimension, segment, alpha_channel) is True:
                                i = i +1
                                box = (i, j, i+segment_dimension, j+segment_dimension)
                                segment = im.crop(box)
                                countback = False
                                break
                    cycle = cycle + 1                                       
                   # segment.save("test_%s" %cycle + ".jpg","JPEG")
                    big_list.extend(alt_save(segment, category))
            else:
                cycle = cycle + 1
                big_list.extend(alt_save(segment, category))
    
    # if you are using the module in standalone mode (i.e. not through ExtractHandler) then you want to implement the following lines of code:
    #output = numpy.array(big_list,numpy.uint8)
    #output.tofile(save_name + ".bin")
    
    print("Image harvest complete: %s images obtained" % cycle)
    
    if cycle is 0:
        print("Bad training file: " + image_name)
    
    return(big_list, cycle) # passes an output to ExtractHandler - remove this line if using module independent of ExtractHandler
    
### end ImageExtractor.harvest


def check_filter(segment_dimension, segment, alpha_channel):

# examines segment to see if it contains the filter colour
# consider implementing this function outside of the iterative loop to improve efficiency, ? use large boolean matrix to identify filter pixels
    
    for n in range(0,segment_dimension,1):
        for m in range(0,segment_dimension,1):
            if segment.getpixel((m,n)) == alpha_channel:
                return True
    
    return False

### end ImageExtractor.check_filter


def alt_save(segment, category):
    
    image_as_array = (numpy.array(segment))
    
    r = image_as_array[:,:,0].flatten()
    g = image_as_array[:,:,1].flatten()
    b = image_as_array[:,:,2].flatten()
    label = [category]
    return(list(label) + list(r) + list(g) + list(b))
    
### end ImageExtractor.alt_save


        