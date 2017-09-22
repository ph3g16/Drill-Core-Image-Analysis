# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:55:07 2017

Superimpose one map onto another.

Use this to filter out unwanted results (from noise or background) or to combine maps of different resolutions.

@author: Peter
"""

def mask(top_layer, base_layer, save_name):
    
    import numpy
    
    # set directories to find/save
    target_folder_top_layer = "Binaries/Boundary_Maps/"
    target_folder_base_layer = "Binaries/12_Window/"
    destination_folder = "Binaries/Output_Maps/"
    
    # unpack data
    stack = numpy.load(target_folder_top_layer + top_layer + ".npy")
    stack2 = numpy.load(target_folder_base_layer + base_layer + ".npy")
    
    # check maps are same size
    print("Top layer height is: {}".format(len(stack)))
    print("Base layer height is: {}".format(len(stack2)))
    if len(stack) != len(stack2):
        print("Maps not suitable for overlay: height is non-identical)")
        return
    if len(stack[0]) != len(stack2[0]):
        print("Maps not suitable for overlay: width is non-identical)")
        return    
    
    # iterate through maps overlaying one if required
    for j in range(len(stack)):
        for i in range(len(stack[j])):
            if stack[j][i] == 255:
                stack2[j][i] = 255     # assign new value to base layer
                    
    # re-pack and save
    
    output = numpy.asarray(stack2)
    numpy.save(destination_folder + save_name + ".npy", output)
    
    print("Map overlay complete, new map saved as: " + save_name + ".npy")   
    
### end Map_Blender.mask

def merge(mode1_layer, mode2_layer, mode3_layer, save_name):
# this function is long because it is hard work to ensure that all three maps agree in size
# to see what it is actually doing once the maps are aligned scroll down to the last 8-10 lines of code
    
    import numpy
    import math
    
    # set directories to find/save
    target_folder_mode1 = "Binaries/Mode1/"
    target_folder_mode2 = "Binaries/Mode2/"
    target_folder_mode3 = "Binaries/Mode3/"
    destination_folder = "Binaries/Output_Maps/"
    
    # unpack data
    stack1 = numpy.load(target_folder_mode1 + mode1_layer + ".npy")
    stack2 = numpy.load(target_folder_mode2 + mode2_layer + ".npy")
    stack3 = numpy.load(target_folder_mode3 + mode3_layer + ".npy")
    
    # load metadata
    load_metadata1 = numpy.load(target_folder_mode1 + mode1_layer + "_meta" + ".npy")
    meta1 = load_metadata1.tolist()
    load_metadata2 = numpy.load(target_folder_mode2 + mode2_layer + "_meta" + ".npy")
    meta2 = load_metadata2.tolist()
    load_metadata3 = numpy.load(target_folder_mode3 + mode3_layer + "_meta" + ".npy")
    meta3 = load_metadata3.tolist()
    
    # check maps are from same source file
    if meta1[2] != meta2[2] or meta1[2] != meta3[2]:
        print("Maps not suitable for merger: original dimensions are not compatible, suggest maps are not drawn from same source file)")
        return  
    if meta1[3] != meta2[3] or meta1[3] != meta3[3]:
        print("Maps not suitable for merger: original dimensions are not compatible, suggest maps are not drawn from same source file)")
        return  
    
    # amend 24x24 map to fit roughly the same scale as 12x12 maps (scale is determined by step_size when converting the core images to binary)
    new_stack = []
    for row in stack1:
        new_row = []
        for element in row:
            for a in range(2):
                new_row.append(element)
        for b in range(2):
            new_stack.append(new_row)
    stack1 = new_stack
    
    # code to reduce map size if required
    """stack3 = stack3[::2]    
    lossy_stack = []
    for element in stack3:
        lossy_stack.append(element[::2])
    stack3 = lossy_stack"""
    
    # adjust 24x24 map to agree with dimensions of 12x12 maps (adjustment required as sampling with different methods will produce very slightly different map widths)
    # do this the listy way....
    
    # fix width
    gap1 = len(stack2[0]) - len(stack1[0])
    if gap1 > 0: # add elements to each column
        for j in range(len(stack1)):
            for i in range(math.floor(gap1/2)):
                stack1[j].insert(0, 255)
            for i in range(math.ceil(gap1/2)):
                stack1[j].append(255)
    elif gap1 < 0: # or delete (via slice)
        for j in range(len(stack1)):
            stack1[j] = stack1[j][math.floor((gap1*-1)/2):len(stack1) - math.ceil((gap1*-1)/2):1]  
            
    # fix height
    empty_row = []
    for element in stack2[0]: empty_row.append(255)
    gap1 = len(stack2) - len(stack1)
    if gap1 > 0: # add rows
        for j in range(math.floor(gap1/2)):
            stack1.insert(0, empty_row)
        for j in range(math.ceil(gap1/2)):
            stack1.append(empty_row)
    elif gap1 < 0: # or delete (via slice)
        stack1 = stack1[math.floor((gap1*-1)/2):len(stack1) - math.ceil((gap1*-1)/2):1]
        
    """
    # .... or the numpy way
    # fix width
    gap1 = len(stack2[0]) - len(stack1[0])
    gap3 = len(stack2[0]) - len(stack3[0])
    if gap1 > 0: # add elements to each column
        for i in range(math.floor(gap1/2)):
            numpy.insert(stack1, 0, 255, axis=1)
        for i in range(math.ceil(gap1/2)):
            numpy.insert(stack1, stack1.shape[1], 255, axis=1)
    elif gap1 < 0: # or delete (via slice)
        for j in range(len(stack1)):
            stack1[j] = stack1[j][math.floor((gap1*-1)/2):len(stack1) - math.ceil((gap1*-1)/2):1]
    if gap3 > 0: # add elements to each column
        for i in range(math.floor(gap3/2)):
            numpy.insert(stack3, 0, 255, axis=1)
        for i in range(math.ceil(gap3/2)):
            numpy.insert(stack3, stack3.shape[1], 255, axis=1)
    elif gap3 < 0: # or delete (via slice)
        for j in range(len(stack3)):
            stack3[j] = stack3[j][math.floor((gap3*-1)/2):len(stack3) - math.ceil((gap3*-1)/2):1]    
            
    # fix height
    gap1 = len(stack2) - len(stack1)
    gap3 = len(stack2) - len(stack3)
    if gap1 > 0: # add rows
        for j in range(math.floor(gap1/2)):
            numpy.insert(stack1, 0, 255, axis=0)
        for j in range(math.ceil(gap1/2)):
            numpy.insert(stack1, stack1.shape[0], 255, axis=0)
    elif gap1 < 0: # or delete (via slice)
        stack1 = stack1[math.floor((gap1*-1)/2):len(stack1) - math.ceil((gap1*-1)/2):1]
    if gap3 > 0: # add rows
        for j in range(math.floor(gap3/2)):
            numpy.insert(stack3, 0, 255, axis=0)
        for j in range(math.ceil(gap3/2)):
            numpy.insert(stack3, stack3.shape[0], 255, axis=0)
    elif gap3 < 0: # or delete (via slice)
        stack3 = stack3[math.floor((gap3*-1)/2):len(stack3) - math.ceil((gap3*-1)/2):1]
    """
    # confirmation for debugging
    print(len(stack1))
    print(len(stack2))
    print(len(stack3))
    print(len(stack1[0]))
    print(len(stack2[0]))
    print(len(stack3[0]))
    
    # ... and now....
    # iterate through maps overlaying one if required
    for j in range(len(stack2)):
        for i in range(len(stack2[j])):
            if stack2[j][i] == 3:
                continue # if we find fault/background in our mode 2 map then skip to the next pixel - fault/background has highest priority
            if stack3[j][i] == 3:
                stack2[j][i] = 3 # also take heed of any fault/background in mode 3 map
                continue
            if stack2[j][i] == 4 or stack2[j][i] == 5 or stack2[j][i] == 9:
                continue # if we find crayon in our mode 2 map then skip to the next pixel - crayon categories are included to prevent eroneous results so make it high priority
            if stack3[j][i] == 4 or stack3[j][i] == 5 or stack3[j][i] == 9:
                stack2[j][i] = stack3[j][i] # the images contain lots of artefacts (even in the black background area!) so want to be very strict on removing eroneous data
                                            # fortunately we have two models that we can trust to supply good data on fault/crayon hence we take information from both models
                continue
            if stack2[j][i] == 8:
                stack2[j][i] = 0 # unfortunately mode 2 finds too much erroneous beige so we opt to remove this
            if stack1[j][i] == 2:
                stack2[j][i] = 6 # add "speckled" data from mode 1
            if stack3[j][i] == 8:
                stack2[j][i] = 8 # now add beige veins from mode 3
            
            # all other data remains untouched (and comes from mode 2)
                  
    # re-pack and save
    output = numpy.asarray(stack2)
    numpy.save(destination_folder + save_name + ".npy", output)
    
    # create a new metafile to accompany the map
    metadata = numpy.asarray(meta2)
    numpy.save(destination_folder + save_name + "_meta" + ".npy", metadata)
    
    print("Map merging complete, new map saved as: " + save_name + ".npy")
    
### end Map_Blender.merge