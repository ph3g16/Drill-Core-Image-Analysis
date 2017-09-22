# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:18:40 2017

Module compares a human map with the machine classified map. Any inconsistencies are judged to be an error on part of the machine.

This is used to validate the final results and evaluate overall accuracy.

Split into a comparison function and a control function which calls the comparison function along with other modules.

@author: Peter
"""

def validate(seg_name, segment_dimension, step_size, mode):
    
    num_categories = 10 # make sure that this is accurate or the CSV file won't write properly!
    
    import Raw_Image_Processor
    Raw_Image_Processor.extract("../Validation/" + seg_name + "_segment", segment_dimension, step_size)
    
    import Map_Generator
    Map_Generator.classify("../Validation/" + seg_name + "_segment", mode)
    
    #import Human_Map
    #Human_Map.map_zones(seg_name + "_zones")
    
    accuracy = compare(seg_name + "_zones", seg_name + "_segment", segment_dimension, step_size, num_categories, seg_name)

    return(accuracy)
    
### end Error_Checking.validate    

def validate_multiple(segment_dimension=12, step_size=6, mode=2):
    
    val_images = ["2A_77_2_1", "2A_90_1_1", "2A_123_2_1"]
    accuracy_figs = []
   
    for val_image in val_images:
        accuracy_figs.append(validate(val_image, segment_dimension, step_size, mode))
    
    print(accuracy_figs)
    
### end Error_Checking.validate_multiple

def compare(human_map, machine_map, segment_dimension, step_size, num_categories, save_name="test"):
    
    import numpy
    import csv
    import math
    
    map_folder = "Binaries/Validation/"
    save_folder = "Results/Validation/"
    
    # unpack data    
    load = numpy.load(map_folder + human_map + ".npy")
    stack = load.tolist()
    stack2 = numpy.load(map_folder + machine_map + ".npy")
    
    # scale up the machine map to rough pixel-equivalence with human map
    new_stack = []
    for row in stack2:
        new_row = []
        for element in row:
            for a in range(step_size):
                new_row.append(element)
        for b in range(step_size):
            new_stack.append(new_row)
    stack2 = new_stack
    new_stack = None
    
    height = len(stack)
    width = len(stack[0])
    
    corrected_height = len(stack2)
    corrected_width = len(stack2[0])
    
    discrepancy = height - corrected_height
    stack = stack[math.floor(discrepancy / 2): height - math.ceil(discrepancy / 2)]
    
    discrepancy = width - corrected_width
    for r in range(len(stack)):
        stack[r] = stack[r][math.floor(discrepancy / 2): width - math.ceil(discrepancy / 2)]
    
    # check maps are same size
    if len(stack) != len(stack2):
        print("Maps not suitable for comparison: height is non-identical")
        return
    if len(stack[0]) != len(stack2[0]):
        print("Maps not suitable for comparison: width is non-identical")
        return    
    

    
    # iterate through maps comparing human classification with machine classification
    # human classification is judged to be "correct" and any disparity is logged as an error
    
    # initialise outcome table:        
    outcomes=numpy.zeros((10, 10), dtype=int)
    # initialise outcome map:
    outcome_map = []      
    
    for j in range(corrected_height):
        map_row=[]
        for i in range(corrected_width):
            human_guess = stack[j][i]
            machine_guess = stack2[j][i]
            # add result to outcomes matrix
            outcomes[machine_guess][human_guess] += 1
            # decide if it is correct or not and add to a new map
            if human_guess == machine_guess:
                map_row.append(0)
            else: map_row.append(1)
        outcome_map.append(map_row)
    
    # calculate precision and recall
    precision = ["Precision"]
    recall = ["Recall"]
    for x in range(num_categories):
        if outcomes[x, x] != 0:
            precision.append(numpy.around( 100 * outcomes[x, x] / sum(outcomes[x]) , decimals=1))
            recall.append(numpy.around( 100 * outcomes[x, x] / sum(outcomes[:,x]) , decimals=1))
        else:
            precision.append(0)
            recall.append(0)
    
    # convert outcome matrix to percentage matrix
    total_pixels = corrected_height * corrected_width
    outcomes = outcomes * 100 / (total_pixels)
    
    # calculate overall accuracy
    accuracy = sum (outcomes[x][x] for x in range(num_categories))
    print("Accuracy: {}".format(numpy.around(accuracy, decimals=1))) 
    
    # round values in outcome matrix for easy viewing
    outcomes = numpy.around(outcomes, decimals=1)
    
    # save results in CSV format
    with open(save_folder + save_name + "_matrix" + ".csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # write headings
        datawriter.writerow(["-", "Background Rock", "Category 1", "Category 2", "Category 3", "Category 4", "Category 5",
                             "Category 6", "Category 7", "Category 8", "Category 9", "Category 10"])
        # add outcome matrix row by row
        outcomes = outcomes.tolist()
        for row in outcomes:
            row_num = outcomes.index(row)
            row.insert(0, "Cat {}".format(row_num))
            datawriter.writerow(row)
        datawriter.writerow(precision)
        datawriter.writerow(recall)
        datawriter.writerow(["Accuracy", numpy.around(accuracy, decimals=1)])
    print("Confusion matrix saved as {}.csv".format(save_folder + save_name + "_matrix"))
    
    # save map
    # repackage outcome comparison as a numpy array and save             
    output = numpy.asarray(outcome_map, numpy.uint8)
    numpy.save(map_folder + save_name + "_comparison" + ".npy", output)
    
    print("Map saved as {}.npy".format(map_folder + save_name + "_comparison"))
    
    return(accuracy)
    
### end Error_Checking.compare