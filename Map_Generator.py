# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:57:07 2017

Load the tensorflow graph, feed in new data, assign predictions to a map and save this as a binary file.

Input data needs to be pre-processed into the cifar10 format. Use Raw_Image_Processor to do this.

Module is written to interact with a modified version of cifar10_eval.py - does not work with original code from tutorial.

@author: Peter
"""

# code is predicated on the assumption that raw data was generated from a rectangular image

def classify(output_file):
    
    import numpy
    import cifar10_eval      # want to hijack functions from the evaluation script
    
    target_folder = "Binaries/"   # finds target file in "Binaries"
    destination_folder = "Binaries/Output_Maps/"   # destination for output file
#    destination_folder = "Binaries/Mode{}/".format(mode)   # alternate destination for output file (use this is performing multi-mode classification)
    target_file = "raw" # since Raw_Image_Processor.extract() is currently coded to output files of this name

    # open the temporary meta file to retrieve x,y dimensions
    file = open(target_folder + target_file + "_meta" + ".txt", "r")
    new_x = int(file.readline())
    new_y = int(file.readline())
    orig_x = int(file.readline())
    orig_y = int(file.readline())
    segment_dimension = int(file.readline())
    step_size = int(file.readline())
    file.close()
    
    # create a new metafile to accompany the map
    metadata = [new_x, new_y, orig_x, orig_y, segment_dimension, step_size]
    numpy.asarray(metadata)
    numpy.save(destination_folder + output_file + "_meta" + ".npy", metadata)
    
    # run cifar10_eval and create predictions vector (formatted as a list)
    predictions = cifar10_eval.map_interface(new_x * new_y)
    
    del predictions[(new_x * new_y):]     # get rid of excess predictions (that are an artefact of the fixed batch size)
    
    print("# of predictions: " + str(len(predictions)))    
    
    # check that we are mapping the whole picture! (evaluation functions don't necessarily use the full data set)
    if len(predictions) != new_x * new_y:
        print("Error: number of predictions from cifar10_eval does not match metadata for this file")
        return
    
    # copy predictions to a nested list to make extraction of x/y data easy
    # also reduces required metadata - x/y dimensions are stored via the shape of the output vector
    # hindsight - it would be a significant improvement to implement this as a numpy array rather than a nested list (all of the code working with map files would need to be amended but would become much shorter)
    stack = []
    for j in range(new_y):
        stack.append([])
        for i in range(new_x):
            stack[j].append(predictions[j*new_x + i])
    
    predictions = None # clear the variable to free up memory

    # we have our final output
    # repackage this as a numpy array and save for later use
             
    output = numpy.asarray(stack,numpy.uint8)
    numpy.save(destination_folder + output_file + ".npy", output)
    
    print("Category mapping complete, map saved as numpy pickle: " + output_file + ".npy")
    
### end Map_Generator.classify

