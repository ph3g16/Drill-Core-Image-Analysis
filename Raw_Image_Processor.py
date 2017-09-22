# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:43:07 2017

Use this to pass arguments to Image_Extractor.py

Processes the raw images into a format which can be read and categorised by the learning engine.

This module differs from Extract_Handler which is intended to process curated/categorised images to generate training data.

@author: Peter
"""

read_directory = "Images/Cores/"
save_directory = "Binaries/"

def extract(file_name, segment_dimension, step_size, HSV=False):
    
    import Image_Extractor
    import numpy
    import math
    import os
    from PIL import Image   
    
    # global variables defining resolution ideally step_size should be a factor of segment_dimension
    # step_size = 6
    # segment_dimension = 24
    
    # delete existing raw image binaries - do this to make sure that we aren't accidentally polluting the data with binaries from a previously processed image
    for f in os.listdir(save_directory):
            if f.startswith("raw_batch"):
                os.remove(save_directory + f)
    
    # create meta-file to hold information about resolution of segmented image
    im = Image.open(read_directory + file_name + ".jpg")  # if processing multiple files from briefcase will need to pop this function into the loop below
    print(im.format,im.size,im.mode)
    x_length, y_length = im.size
    im.close()
    
    new_x = math.ceil((x_length - segment_dimension+1) / step_size)
    new_y = math.ceil((y_length - segment_dimension+1) / step_size)
    
    metadata = [new_x, new_y, x_length, y_length, segment_dimension, step_size]
    
    file = open(save_directory + "raw" + "_meta" + ".txt", "w")
    for element in metadata:
        file.write(str(element) + "\n")    # writes next aspect of metadata
    file.close()
    print("Metadata saved as: " + "raw" + "_meta" + ".txt")

    # calculate slice size and overlap between slices
    # all values given in terms of pixels from original image (e.g. slice_height is the number of pixels from the original image that will be processed per slice)
    
    slice_height = (math.floor((30112 / new_x)-1) * step_size) + segment_dimension    # round down to avoid memory overload
#    slice_height = int(slice_height - (slice_height % (segment_dimension/step_size)))    # round down such that the slice height makes sense in terms of image segmentation

    overlap = segment_dimension - step_size
    num_slices = math.ceil(y_length / (slice_height - overlap))
    print("Chopping data into {} portions.".format(num_slices))
    
    if slice_height < segment_dimension*2 :   # rudimentary formula - may need to modify
        print("Error: not able to extract data of this resolution. Consider modifying code to slice data vertically as well as horizontally.")
        return
    
    # iterate through the raw data image passing slice dimensions to Image_Extractor
    # parcels data according to slice number
    for portion in range(num_slices):        
        stack, recordcount = Image_Extractor.harvest(read_directory + file_name + ".jpg",
                                        0, False, segment_dimension, step_size, limit=1000000,
                                        portion=[0, portion*(slice_height-overlap), x_length, min((portion*(slice_height-overlap))+slice_height, y_length)],
                                        HSV=HSV)     # assign each sub-image category zero - this isn't used for categorisation but allows us to re-use the script to encode/decode binaries
        
        output = numpy.array(stack,numpy.uint8)
        output.tofile(save_directory + "raw_batch" + "_" + str(portion+1).zfill(3) + ".bin")
        
        print("Extract successful, slice saved as: " + "raw_batch" + "_" + str(portion+1).zfill(3) + ".bin")
     
    
### end ExtractHandler.extract