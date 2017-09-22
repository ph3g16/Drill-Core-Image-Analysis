# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:32:15 2017

Module for analysing a map and extracting quantified results.

This is used to assess non-planar features. Use Sine_Analysis.py to assess and quantify planes.

Module is currently set up to process up to 10 categories. Excess categories will be treated as background/empty.

@author: Peter
"""
import numpy
import csv
import os
import math

num_categories = 20

def horizontal(file_name, block_size=30, graph=True):

# defaults to assessing 30 rows at a time
# graph specifies whether you are using the function to generate visualisation or quantification
    
    import matplotlib.pyplot as plt
    
    target_folder = "Binaries/Output_Maps/"
    
    # unpack data
    load_array = numpy.load(target_folder + file_name + ".npy")      
    stack = load_array.tolist()
    
    # check that block size is smaller than image height
    if len(stack) < block_size:
        return("Map too small")
    
    # iterate through blocks to assess percentage involvement of each category   
    
    spacing = 1         # this sets the program to move the block 1 row down each time
    percentages = []
    num_blocks = len(stack) - block_size + spacing
    map_width = len(stack[0])
    data_points_per_block = map_width * block_size
    
    for block in range(num_blocks):
        # set count variables
        cat_zero = 0
        cat_one = 0
        cat_two = 0
        cat_three = 0
        cat_four = 0
        cat_five = 0
        cat_six = 0
        cat_seven = 0
        cat_eight = 0
        cat_nine = 0
        
        for row in range(block, block_size + block, 1):
            for element in stack[row]:
                if element is 0:
                    cat_zero += 1
                elif element is 1:
                    cat_one += 1
                elif element is 2:
                    cat_two += 1
                elif element is 3:
                    cat_three += 1
                elif element is 4:
                    cat_four += 1
                elif element is 5:
                    cat_five += 1
                elif element is 6:
                    cat_six += 1
                elif element is 7:
                    cat_seven += 1
                elif element is 8:
                    cat_eight += 1
                elif element is 9:
                    cat_nine += 1
                else:
                    print("Error, a map element does not correspond to one of the anticipated categories. If your data has more than 10 categories you need to edit this module.")

        # calculate percentages
        category_percentages = [cat_zero, cat_one, cat_two, cat_three, cat_four,
                                    cat_five, cat_six, cat_seven, cat_eight, cat_nine]
        for category in range(len(category_percentages)):
            category_percentages[category] = (category_percentages[category] / data_points_per_block) * 100
        
        # append results as a sublist of final output
        percentages.extend(category_percentages)
    
    if graph:
        # generate a graph
        
        # first assemble a series of sets to plot
        percentages_zero = numpy.asarray(percentages[0:None:10])
        percentages_one = numpy.asarray(percentages[1:None:10]).T
        percentages_two = percentages[2:None:10]
        percentages_three = percentages[3:None:10]
        percentages_four = percentages[4:None:10]
        percentages_five = percentages[5:None:10]
        percentages_six = percentages[6:None:10]
        percentages_seven = percentages[7:None:10]
        percentages_eight = percentages[8:None:10]
        percentages_nine = percentages[9:None:10]
        
        fig, ax = plt.subplots()
        
        line1, = ax.plot(percentages_zero, linewidth=1, label="Plot of Category 0")
        line2, = ax.plot(percentages_one, linewidth=1, label="Plot of Category 1")
        line3, = ax.plot(percentages_two, linewidth=1, label="Plot of Category 2")
        line4, = ax.plot(percentages_three, linewidth=1, label="Plot of Category 3")
        line5, = ax.plot(percentages_four, linewidth=1, label="Plot of Category 4")
        line6, = ax.plot(percentages_five, linewidth=1, label="Plot of Category 5")
        line7, = ax.plot(percentages_six, linewidth=1, label="Plot of Category 6")
        line8, = ax.plot(percentages_seven, linewidth=1, label="Plot of Category 7")
        line9, = ax.plot(percentages_eight, linewidth=1, label="Plot of Category 8")
        line10, = ax.plot(percentages_nine, linewidth=1, label="Plot of Category 9")
        
    #    y = slice(0, num_blocks)
    #    x = slice(percentages[0:None, 10])
    #    line1, = ax.plot(x, y, linewidth=1, label="Plot of Category 0")
        
        plt.xlabel('Block Number')
        plt.ylabel('Percentage')
        plt.title('% by block, block size {}'.format(block_size))
        ax.legend()
        ax.grid(True)
    
        plt.show()
        
    else:
        # if using function for feature analysis:
        return(percentages)
    
### end Map_Analysis.horizontal

def feature_count(file_name, category, threshold):
    
    # do horizontal feature analysis
    percentages = horizontal(file_name, block_size=20, graph=False)
    
    # bin analysis for features we aren't interested in
    percentages = percentages[category:None:10]

    # iterate through data to find interesting features
    feature = False
    for element in percentages:
        if element > 20:    # if classification density is greater than threshold then there must be a feature
            if not feature:
                # mark start of feature
                pass
            feature = True            
        else:               # if classification density is lower than threshold then there is no feature, mark end of feature if there was one in the previous block
            if feature:
                # mark end of feature
                pass
            feature = False        

### end Map_Analysis.feature_count

def quantify(file_name, by_line=True):
# count how many times a given feature appears within a map on a line by line basis
# perform the count line by line as it is then easy to reconstruct as Xmm height blocks
    
    # set directories to find/save
    
    target_folder = "Binaries/Output_Maps/"
    save_folder = "Results/Quantities/"
    
    # unpack data      
    load_array = numpy.load(target_folder + file_name + ".npy")      
    stack = load_array.tolist()
    
    if by_line: suffix = "_quant_by_line"
    else: suffix = "_quantities"
    
    with open(save_folder + file_name + suffix + ".csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # write headings
        datawriter.writerow(["Cat {}".format(i) for i in range(num_categories)] + ["None"])
        
        # initialise counter then enumerate the number of times the category appears in the rock
        count = [0 for _ in range(num_categories)] # create an empty array of size num_categories
        empty = 0
        
        for j in range(len(stack)):
            if by_line:
                # reset the counters to start a new line
                count = [0 for _ in range(num_categories)]
                empty = 0
            row = stack[j]
            for i in range(len(stack[j])):
                try: count[row[i]] += 1
                except IndexError: empty += 1
            if by_line: datawriter.writerow(count + [empty])
        
        # if recording quantities en mass we still need to write them to a file
        if not by_line:
            datawriter.writerow(count + [empty])

### end Map_Analysis.quantify()

def collate_quantifications():
# open all "by_line" CSV files in target folder, aggregate them in some way and save the results to a single CSV
# this could be done more conveniently by putting the initial data into one large excel file and editing from that - unfortunately there is a limit of 1 million rows which is just slightly too few

    # set variables
    target_folder = "Results/Quantities/"
    step_size = 6 # number of pixels jumped between classifications
    pixel_size = 0.0001 # each pixel represents 1/10th of a mm
    rows_to_sum = 16
    output_frequency = 8 # calculate a summation every x rows
    
    category_height = step_size * pixel_size # category_height is the vertical distance covered by each row of classifications
    sum_distance = round(category_height * rows_to_sum, 4) # sum_distance is the vertical distance over which each sum is taken
    distance_seperating_sums = round(category_height * output_frequency, 4) # e.g. if each row represents 1mm and we calculate a sum every 10 rows then we will be taking a meaurement every 1cm
    
    # create master file to dump collated results into
    with open(target_folder + "Density_per_segment" + ".csv", 'w', newline='') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # write headings
            datawriter.writerow(["Depth"] + ["Cat {}".format(i) for i in range(num_categories)])
    
    # read depth file and create dictionary
    depth_index = {}
    with open("Results/" + "2A_Depth_Index" + ".csv", newline='') as depth_csv:
        # setup reader function
        datareader = csv.reader(depth_csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        while True:
            try:
                row = next(datareader)
                depth_index[row[1]] = float(row[0]) # for each row add a depth value to the dictionary which can be accessed via the core name
            except StopIteration:
                break
    
    file_list = os.listdir(target_folder)
    file_list.sort()
    for csv_file in file_list:
        # ensure that we are only fetching "by_line" files
        if csv_file[-11:] != "by_line.csv": continue
        stack = []
        core_name = csv_file[0:19]
        # fetch starting depth
        if core_name in depth_index:
            depth = depth_index[core_name]
        else:
            print("Depth value for {} is missing".format(core_name))
            depth = -5 # assign arbitrary value
        # open file and unpack the data
        with open(target_folder + csv_file, newline='') as csvfile:
            # setup reader function
            datareader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # skip headings
            next(datareader)
            # fetch first line of quantities and start a loop which cycles through all rows in the file
            while True:
                try:
                    row = next(datareader)
                    stack.append(row)
                except StopIteration:
                    break
        # push data into a numpy array to make it easy to reference
        stack = numpy.asarray(stack)
        stack = stack.astype(int)
        # calculate local summation and attach a depth value to each
        rolling_average = []
        height, width = stack.shape
        for y in range(math.floor(rows_to_sum/2), height-math.ceil(rows_to_sum/2), output_frequency):
            depth = round(depth + distance_seperating_sums, 4)
            quantities = [numpy.sum(stack[y-math.floor(rows_to_sum/2):y+math.ceil(rows_to_sum/2), i]) for i in range(num_categories)]
            # at this stage we could stop and just report the quantity per cm
            # however, it makes more sense to report figures in terms of proportion of rock (e.g. to allow easier comparison between cores of different sizes)
            artefact_quantity = quantities[0] + quantities[2] + quantities[3] + quantities[4]   # define background and blue/red/yellow crayon as artefacts
            rock_quantity = sum(quantities) - artefact_quantity
            if rock_quantity > 500:
                densities = [quantities[i]/rock_quantity for i in range(num_categories)]
            else: densities = [0 for _ in range(num_categories)]
            densities.insert(0, depth)
            rolling_average.append(densities)
    
        # append results to master file we created earlier
        with open(target_folder + "Density_per_segment" + ".csv", 'a', newline='') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # write rows            
            for row in rolling_average:
                datawriter.writerow(row)
    
    # declare success
    print("Data collated in {}m chunks reported every {}m".format(sum_distance, distance_seperating_sums))
    print("Results saved to " + target_folder + "Density_per_segment" + ".csv")

### end Map_Analysis.collate_quantifications