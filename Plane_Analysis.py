# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:38:41 2017

Module for analysing a map and identifying planes passing through the cylinder.

Since these are maps of an unrolled cylinder planes manifest themseves as sine curves.
Code attempts to identify possible planes and then find a best fit for each.

Main function is find_all_curves() which opens a map and attempts to find every flat non-background plane.
The final set of planes is saved to a CSV file.

@author: Peter
"""

import numpy
import math

# loss is used to speed up calculations
# increase loss as required (higher resolution = require higher loss)
loss = 4   

def find_all_curves(file_name):
    
    import csv

    target_folder = "Binaries/Output_Maps/"
    save_folder = "Results/Curves/"
    
    # unpack data
    load_array = numpy.load(target_folder + file_name + ".npy")      
    stack = load_array.tolist()
    
    # create low-res copy 
    crude_stack = bin_data(stack, loss)
    
    map_width = len(stack[0])
    map_height = len(stack)
    
    curveset = []
    
    # set categories to seach for and the threshold at which to search
    categories = [[1, 0.5], [6, 0.5], [7, 0.25], [8, 0.5]]
    
    # find the curves (will take some time - anticipate 4 mins per category)
    for category in categories:
        curveset.extend( fit_sine_curve(stack, crude_stack, category[0], initial_threshold=category[1]) )
    
    # add accuracy statistic and determine how much of the wave is present in the sample
    for curve in range(len(curveset)):
        curveset[curve].extend( ascertain_accuracy(stack, curveset[curve], map_width, map_height) )
    
    # add angle measurement
    for curve in range(len(curveset)):
        curveset[curve].append( calculate_slope(curveset[curve]) )
    
    # save in CSV format
    with open(save_folder + file_name + "_curves" + ".csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # write headings
        datawriter.writerow(["Amplitude", "Displacement", "Thickness", "Y Intercept", "Fitness", "Category", "Accuracy", "% Capture", "Angle"])
        # add curve data
        for curve in curveset:
            datawriter.writerow(curve)
    
### end Plane_Analysis.find_all_curves

def bin_data(stack, loss_ratio):
# reduces the map resolution by binning a huge amount of data (if loss ratio=2 it bins 3/4 of the data)
# this makes it cheaper to analyse and thus is useful if speed is more important than precision 
 
    stack = stack[::loss_ratio]
    
    lossy_stack = []
    for element in stack:
        lossy_stack.append(element[::loss_ratio])
    
    return(lossy_stack)
    
### end Plane_Analysis.bin_data

def assess_fitness(stack, map_width, map_height, amplitude, displacement, thickness, y_intercept, category):
    
    count = 0
    fit = 0
    # assign x/y coordinates for sine wave and produce fitness statistic
    for sin_x in range(map_width):
        sin_y = int(amplitude * math.sin(((sin_x + displacement) * 2 * math.pi) / map_width)) + y_intercept
        for y_coord in range(sin_y, sin_y + thickness):   
            compare = True
            # check that y-coordinate is possible to reference and has not been assigned as background
            if y_coord >= map_height:
                compare = False
            if y_coord < 0:
                compare = False
            # assuming we have a sensible y-coordinate check to see if it contains the category we are interested in
            if compare:
                if stack[y_coord][sin_x] is not 255:
                    count += 1
                    if stack[y_coord][sin_x] is category:
                        fit += 1
    accuracy = fit / max(count, map_width/4)
    # notice that the function is designed to be able to evaluate partial sine waves which have been cut off at the top or the bottom
    # max(count, map_width/4) is used to put a minimum floor on the number of pixels which can register as a sine wave
    return(accuracy)

### end Plane_Analysis.assess_fitness

def draw_sine_curve(stack, map_width, map_height, curve):
    
    amplitude, displacement, thickness, y_intercept = curve[0], curve[1], curve[2], curve[3]
    # assign x/y coordinates and draw upper limit of wave
    for sin_x in range(0, map_width, 2):
        sin_y = int(amplitude * math.sin(((sin_x + displacement) * 2 * math.pi) / map_width)) + y_intercept
        if -1 < sin_y < map_height:
            stack[sin_y][sin_x] = 5
    # assign x/y coordinates and draw lower limit of wave
    for sin_x in range(0, map_width, 2):
        sin_y = int(amplitude * math.sin(((sin_x + displacement) * 2 * math.pi) / map_width)) + y_intercept + (thickness - 1)
        if -1 < sin_y < map_height:
            stack[sin_y][sin_x] = 5
            
    return(stack)
    
### end Plane_Analysis.draw_sine_curve

def ascertain_accuracy(stack, curve, map_width, map_height):
# duplicates a chunk of code from the assess_fitness function

    # unpack curve elements
    amplitude, displacement, thickness, y_intercept, category = curve[0], curve[1], curve[2], curve[3], curve[5]

    pixel_count = 0
    empty_count = 0
    # assign x/y coordinates for sine wave and produce fitness statistic
    for sin_x in range(map_width):
        sin_y = int(amplitude * math.sin(((sin_x + displacement) * 2 * math.pi) / map_width)) + y_intercept
        for y_coord in range(sin_y - 20, sin_y + thickness + 20):   
            compare = True
            # check that y-coordinate is possible to reference and has not been assigned as background
            if y_coord >= map_height:
                compare = False
            if y_coord < 0:
                compare = False
            # assuming we have a sensible y-coordinate check to see if it contains the category we are interested in
            if compare:
                if stack[y_coord][sin_x] is category:
                    pixel_count += 1
    
    # simple calculation showing the area that you think the sine wave covers
    sine_wave_area = map_width * thickness
    
    # calculate area that you would expect given that some curves are cut off by the edge of the picture
    for sin_x in range(map_width):
        sin_y = int(amplitude * math.sin(((sin_x + displacement) * 2 * math.pi) / map_width)) + y_intercept
        for y_coord in range(sin_y, sin_y + thickness):   
            compare = True
            # check that y-coordinate is possible to reference and has not been assigned as background
            if y_coord >= map_height:
                compare = False
            if y_coord < 0:
                compare = False
            # assuming we have a sensible y-coordinate check to see if it contains the category we are interested in
            if compare:
                if stack[y_coord][sin_x] is 255:
                    empty_count += 1
            else: empty_count += 1
    
    actual_area = sine_wave_area - empty_count
    
    # turn this into a decimal accuracy (decimal > 1 indicates that the sine wave is underestimating the interested area)
    accuracy = pixel_count / actual_area
    portion_of_wave_captured = actual_area / sine_wave_area
    
    return([accuracy, portion_of_wave_captured])
    
### end Plane_Analysis.ascertain_accuracy

def calculate_slope(curve):
# iterates through a set of curves and returns the incline of each curve
# slope calculation depends on radius of cylinder and amplitude of wave
    
    radius = 31.75 # mm
    # other radii are 48 (large borehole), 23.8 (smaller cores), 37.85 (smaller borehole)
    amplitude = curve[0]/10 # estimated on the assmption of 10 pxels per mm
    #angle from horizontal
    angle = math.atan(radius/amplitude)
    #angle from vertical
    angle = math.atan(amplitude/radius)
    
    return(angle)
        
### end Plane_Analysis.calculate_slope

def fit_sine_curve(stack, crude_stack, category, initial_threshold=0.5, file_name="optional"):
    
    if file_name == "optional":
        testing = False
    else: testing = True
        
    if testing:
        target_folder = "Binaries/Maps/"
        # unpack data
        load_array = numpy.load(target_folder + file_name + ".npy")      
        stack = load_array.tolist()
        crude_stack = bin_data(stack, loss)
    
    map_width = len(stack[0])
    map_height = len(stack)

    crude_width = len(crude_stack[0])
    crude_height = len(crude_stack)
    # find all sine curves with fit stronger than threshold
    curves = []
    skip_y = 0 # variables used to cut down the number of duplicate curves
    y_range = range(0, crude_height, 2)
    amp_range = range(0, 200, 4)
    disp_range = range (0, crude_width, 3)
    for test_y in y_range:
        if skip_y != 0:
            skip_y -= 1
            continue
        skip_amp = 0
        for test_amp in amp_range:
            if skip_amp != 0:
                skip_amp -= 1
                continue
            skip_disp = 0
            for test_disp in disp_range:
                if skip_disp != 0:
                    skip_disp -= 1
                    continue
                fitness = assess_fitness(crude_stack, crude_width, crude_height, test_amp, test_disp, 2, test_y, category)
                if fitness > initial_threshold:
                    curves.append([test_amp, test_disp, 2, test_y])
                    skip_y, skip_amp, skip_disp = 2, 2, 2  # skip the next x,y,z many cycles for y, amplitude and displacement, used to reduce duplication

    print(len(curves))
    
    # reshape curves to get better fit
    # rough fitting is also the primary way to add a thickness element to each curve
    for curve in range(len(curves)):
        curves[curve] = rough_fit(crude_stack, crude_width, crude_height, curves[curve], category)
    
    # find the thickest curve
    # eliminate any other curves that share a substantial cross-section with this curve
    # set the thickest curve to one side, find the next thickest and repeat until there are no remaining curves with substantial intersection    
    unique_curves = []
    for repeat in range(2000):
        if not curves: break # if curves is empty then it will returne a boolean: False
                             # this loop should run until curves is empty
        duplicates = []
        # find thickest curve
        thickest = max(curve[2] for curve in curves)
        fatty = next((i for i, sublist in enumerate(curves) if thickest in sublist), -1)
        fatty_coords = []
        fatty_thickness = curves[fatty][2]
        for sin_x in range(map_width):
            sin_y = int(curves[fatty][0] * math.sin(((sin_x + curves[fatty][1]) * 2 * math.pi) / map_width)) + curves[fatty][3]        
            fatty_coords.append(sin_y)
        unique_curves.append(curves[fatty])
        del curves[fatty]
        # compare the other curves, see which are subsumed
        for c in range(len(curves)):
            crossover = 0
            for sin_x in range(map_width):
                sin_y = int(curves[c][0] * math.sin(((sin_x + curves[c][1]) * 2 * math.pi) / map_width)) + curves[c][3]
            
                for coord in range(sin_y, sin_y + curves[c][2], 1):
                    
                    if fatty_coords[sin_x]-1 < coord < fatty_coords[sin_x] + fatty_thickness:
                        crossover += 1
                
            # if there is too much crossover then mark the curve for deletion
            if crossover / (map_width * curves[c][2]) > 0.03:
                duplicates.append(c)
        duplicates = duplicates[::-1]
        for d in duplicates:
            del curves[d]
   
    curves = unique_curves
    
    # resize remaining curves to fit the real data (rather than the low res temp data used to find the curves)
    for curve in range(len(curves)):
        curves[curve][0] = curves[curve][0] * loss
        curves[curve][1] = curves[curve][1] * loss
        curves[curve][2] = curves[curve][2] * loss
        curves[curve][3] = curves[curve][3] * loss
        
    # at this stage we should have a modest number of thick curves which are pretty much independent from each other
    # given that there should be a small number we can afford to dedicate resources to fine tuning the fit
    for curve in range(len(curves)):
        curves[curve] = refine_fit(stack, map_width, map_height, curves[curve], category)
    
    for curve in range(len(curves)):
        curves[curve].append(category)
    
    # add the curves to the map and display
    temp_stack = stack
    for curve in curves:
        temp_stack = draw_sine_curve(temp_stack, map_width, map_height, curve)
    
    if testing:
        output = numpy.asarray(temp_stack)
        numpy.save("Binaries/Maps/" + "test_analysis" + ".npy", output)    
        print("Sine drawing complete, new map saved as: " + "test_analysis" + ".npy")
    
        import Map_Viewer
        Map_Viewer.view(None, has_metadata=False, save=False, direct_feed=temp_stack)

    print(curves)    
    return(curves)
    
### end Plane_Analysis.fit_sine_curve

def rough_fit(crude_stack, crude_width, crude_height, curve, category):
# use the low-res map to perform initial fitting
    
    amplitude, displacement, thickness, y_intercept = curve[0], curve[1], curve[2], curve[3]

    # vary thickness  (crude)    
    possibilities = range(thickness, thickness + 150, 2)
    results = []
    for variable in possibilities:
        adjusted_fitness = ((thickness+1)**0.3) * assess_fitness(crude_stack, crude_width, crude_height, amplitude, displacement, variable, y_intercept, category)
        results.append(adjusted_fitness)    
    # search through the list to see which curve fitted best, use adjusted thickness to bias towards thicker bands
    thickness = possibilities[results.index(max(results))]
    
    """# vary y_intercept (crude)   
    possibilities = range(y_intercept - 8, y_intercept + 8, 2)
    results = []
    for variable in possibilities: results.append( assess_fitness(crude_stack, crude_width, crude_height, amplitude, displacement, thickness, variable, category) )    
    # search through the list to see which curve fitted best
    y_intercept = possibilities[results.index(max(results))]
    
    # vary amplitude (crude)
    possibilities = range(amplitude - 100, amplitude + 100, 20)
    results = []
    for variable in possibilities: results.append( assess_fitness(crude_stack, crude_width, crude_height, variable, displacement, thickness, y_intercept, category) )    
    # search through the list to see which curve fitted best
    amplitude = possibilities[results.index(max(results))]
    
    # vary displacement (crude)   
    possibilities = range(displacement - 30, displacement + 30, 6)
    results = []
    for variable in possibilities: results.append( assess_fitness(crude_stack, crude_width, crude_height, amplitude, variable, thickness, y_intercept, category) )    
    # search through the list to see which curve fitted best
    displacement = possibilities[results.index(max(results))]"""
    
    return([amplitude, displacement, thickness, y_intercept])
    
### end Plane_Analysis.crude_fit


def refine_fit(stack, map_width, map_height, curve, category):
    
    amplitude, displacement, thickness, y_intercept = curve[0], curve[1], curve[2], curve[3]
    
    ## attempt to fit curve to data
    
    # vary y_intercept    
    possibilities = range(y_intercept - 6, y_intercept + 6, 2)
    results = []
    for variable in possibilities: results.append( assess_fitness(stack, map_width, map_height, amplitude, displacement, thickness, variable, category) )    
    # search through the list to see which curve fitted best
    y_intercept = possibilities[results.index(max(results))]
    
    # vary displacement    
    possibilities = range(displacement - 40, displacement + 40, 2)
    results = []
    for variable in possibilities: results.append( assess_fitness(stack, map_width, map_height, amplitude, variable, thickness, y_intercept, category) )    
    # search through the list to see which curve fitted best
    displacement = possibilities[results.index(max(results))]
    
    # vary amplitude    
    possibilities = range(amplitude - 20, amplitude + 20, 2)
    results = []
    for variable in possibilities: results.append( assess_fitness(stack, map_width, map_height, variable, displacement, thickness, y_intercept, category) )    
    # search through the list to see which curve fitted best
    amplitude = possibilities[results.index(max(results))]
    
    # vary thickness    
    possibilities = range(thickness, thickness + 20, 2)
    results = []
    for variable in possibilities:
        adjusted_fitness = (thickness + 1) * assess_fitness(stack, map_width, map_height, amplitude, displacement, variable, y_intercept, category)
        results.append(adjusted_fitness)    
    # search through the list to see which curve fitted best, use adjusted thickness to bias towards thicker bands
    thickness = possibilities[results.index(max(results))]
    
    # vary y_intercept    
    possibilities = range(y_intercept - 20, y_intercept + 20, 2)
    results = []
    for variable in possibilities: results.append( assess_fitness(stack, map_width, map_height, amplitude, displacement, thickness, variable, category) )    
    # search through the list to see which curve fitted best
    y_intercept = possibilities[results.index(max(results))]
    
    ## fine tuning  
    
    # tune displacement    
    possibilities = range(displacement - 3, displacement + 3, 1)
    results = []
    for variable in possibilities: results.append( assess_fitness(stack, map_width, map_height, amplitude, variable, thickness, y_intercept, category) )    
    # search through the list to see which curve fitted best
    displacement = possibilities[results.index(max(results))]
    
    # tune amplitude    
    possibilities = range(amplitude - 3, amplitude + 3, 1)
    results = []
    for variable in possibilities: results.append( assess_fitness(stack, map_width, map_height, variable, displacement, thickness, y_intercept, category) )    
    # search through the list to see which curve fitted best
    amplitude = possibilities[results.index(max(results))]
        
    # tune thickness and y_intercept in conjunction   
    thick_possibilities = range(max(thickness - 3, 1), thickness + 3, 1)
    y_possibilities = range(y_intercept - 3, y_intercept + 3, 1)
    results = []
    for y_var in y_possibilities:
        for thick_var in thick_possibilities:
            adjusted_fitness = (thickness + 1) * assess_fitness(stack, map_width, map_height, amplitude, displacement, thick_var, y_var, category)
            results.append(adjusted_fitness)    
    # search through the list to see which curve fitted best, use adjusted thickness to bias towards thicker bands
    best_fitness_index = results.index(max(results))
    thickness = thick_possibilities[best_fitness_index % (len(thick_possibilities))]
    y_intercept = y_possibilities[math.floor(best_fitness_index / len(thick_possibilities))]
    
    fitness = assess_fitness(stack, map_width, map_height, amplitude, displacement, thickness, y_intercept, category)

    return([amplitude, displacement, thickness, y_intercept, fitness])

### end Plane_Analysis.improve_fit