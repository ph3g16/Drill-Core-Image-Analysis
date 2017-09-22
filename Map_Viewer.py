# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:30:53 2017

Takes a category map and outputs visual representation.

@author: Peter
"""

# outputs primary colours (one per category)
# representation corresponds to shape of raw image but does not have same resolution
# use this to visualise data as categories

# need seperate program if want to save categories from original image as layers

def view(file_name, has_metadata=True, save=True, direct_feed=None, show_im=True):
# view = True if want to immediately view the map
# expand = x*x pixel area you want to assign to each prediction

    import numpy
    from PIL import Image
    
    # set directories to find/save
    target_folder = "Binaries/Output_Maps/"
    destination_folder = "Images/Maps/"
    
    # unpack data
    if direct_feed is None:
        load_array = numpy.load(target_folder + file_name + ".npy")      
        stack = load_array.tolist()
    else: stack = direct_feed
    
    # assign category colours
    background = [1, 1, 1]
    colour0 = [0, 0, 0]    # black - background
    colour1 = [0, 255, 0]     # ugly green (bright) - rock
    colour2 = [255, 0, 255]   # dull pink - red crayon
    colour3 = [164, 0, 164]   # purple - blue crayon
    colour4 = [255, 255, 0]   # yellow - yellow crayon
    colour5 = [255, 0, 0]     # bright red - laumontite
    colour6 = [0, 0, 255]     # blue - albetised
    colour7 = [128, 128, 0]   # olive - olivine
    colour8 = [255, 192, 0]   # orange - epidote
    colour9 = [0, 255, 255]   # cyan - altered gabbro
    colour10 = [159, 14, 14]  # dark red - oxidation
    colour11 = [128, 128, 128] # grey - plagiogranite
    colour12 = [9, 198, 143]  # teal - clinozoisite
    colour13 = [51, 181, 0]   # darker green - anhydrite
    colour14 = [72, 72, 72]  # dark grey - prehnite
    colour15 = None   # dark grey - quartz
    colour16 = None
    colour17 = None
    colour18 = None
    colour19 = None

    # iterate through map list and replace category labels with RGB colours
    for j in range(len(stack)):
        row = stack[j]
        for i in range(len(row)):
            if row[i] is 255:
                row[i] = background
            elif row[i] is 0:
                row[i] = colour0
            elif row[i] is 1:
                row[i] = colour1
            elif row[i] is 2:
                row[i] = colour2
            elif row[i] is 3:
                row[i] = colour3
            elif row[i] is 4:
                row[i] = colour4
            elif row[i] is 5:
                row[i] = colour5
            elif row[i] is 6:
                row[i] = colour6
            elif row[i] is 7:
                row[i] = colour7
            elif row[i] is 8:
                row[i] = colour8
            elif row[i] is 9:
                row[i] = colour9
            elif row[i] is 10:
                row[i] = colour10
            elif row[i] is 11:
                row[i] = colour11
            elif row[i] is 12:
                row[i] = colour12
            elif row[i] is 13:
                row[i] = colour13
            elif row[i] is 14:
                row[i] = colour14
            elif row[i] is 15:
                row[i] = colour15
            elif row[i] is 16:
                row[i] = colour16
            elif row[i] is 17:
                row[i] = colour17
            elif row[i] is 18:
                row[i] = colour18
            elif row[i] is 19:
                row[i] = colour19
        stack[j] = row
    
    # load metadata
    if has_metadata:
        load_metadata = numpy.load(target_folder + file_name + "_meta" + ".npy")      
        metadata = load_metadata.tolist()
        
        # expand category areas to form pixel equivalence between map and original image
        stack = map_classifications_to_larger_area(stack, metadata)
        
        # add border
        # not implemented in final version - crop is performed on the original image instead
        
    # convert back to array (because Pillow likes this)
    output = numpy.asarray(stack, numpy.uint8)  
    # convert into image format
    im = Image.fromarray(output)
    if save:
        im.save(destination_folder + file_name + ".png")
        #    im.save(destination_folder + file_name + ".jpg", format='JPEG', subsampling=0, quality=100)    # settings to save a lossless JPEG   
        print("Image saved as " + file_name + ".png in " + destination_folder) 

    if show_im:
        im.show() 

# end Map_Viewer.save              


def map_classifications_to_larger_area(stack, metadata):
    
    # iterate through map list and explode each category to cover more pixels
    # assigns a step_size x step_size area to each classification input to achieve correspondance with original image
    step_size = metadata[5]
    new_stack = []
    for row in stack:
        new_row = []
        for element in row:
            for a in range(step_size):
                new_row.append(element)
        for b in range(step_size):
            new_stack.append(new_row)
    stack = new_stack
    print(len(stack))
    new_stack = None
    new_row = None     # clear the variables to free up memory
    return(stack)

### end Map_Viewer.map_classifications_to_larger_area

def add_border(stack, metadata):
    
# add a border to the image to indicate that some information has been lost
# border also ensures that map has 1-1 correspondance with original image which makes processing easier

    """this code is broken and is not currently implemented - consider replacing it with code that applies a border using pillow and im.crop"""
    
    # extract metadata
    new_x, new_y, orig_x, orig_y, segment_dimension, step_size = metadata[0], metadata[1], metadata[2], metadata[3], metadata[4], metadata[5]

    # calculate border dimensions
    top_and_left_thickness = int((segment_dimension - step_size) / 2)
    right_thickness = int(top_and_left_thickness + (orig_x - (top_and_left_thickness * 2 + step_size * new_x)))
    bottom_thickness = int(top_and_left_thickness + (orig_y - (top_and_left_thickness * 2 + step_size * new_y)))
    
    print(top_and_left_thickness)
    print(right_thickness)
    print(bottom_thickness)
    
    print(len(stack[0]))
    # add the right then left borders
    for row in stack:
        for b in range(right_thickness):
            row.append(255)
        for b in range(top_and_left_thickness):
            row.insert(0, 255)
    print(len(stack[0]))

    # add the top and bottom borders
    row = []
    for i in range(len(stack[0])):
        row.append(255)          # create a blank row
    for b in range(top_and_left_thickness):
        stack.insert(0, row)    # append the blank row to the top x many times
    for b in range(bottom_thickness):
        stack.append(row)       # append the blank row to the bottom of the map
 
    return(stack)

### end Map_Viewer.add_border

def test():
# create a sample map to test the viewer with
# generates a 3x3 map with two categories in the shape of a T 
    
    import numpy
    
    destination_folder = "Binaries/Maps/"
    
    stack = [     #metadata layer
             [0,0,0],
             [1,0,1],
             [1,0,1],
             ]
    
    print(stack)
    output = numpy.array(stack)
    print(output)
    numpy.save(destination_folder + "test_map" + ".npy", output)


            