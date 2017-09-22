# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 17:56:27 2017

Module will generate a series of examples.
- Each example should be assessed by a geologist who will input a label
- Input labels are compared against the machine classifications to generate an accuracy figure

Examples are picked based on machine classifications. This should generate a good mix of classes
rather than being biased towards showing examples of whichever feature is most common.

Module now allows user to load a set of examples which was classified previously.
This allows 2+ geologists to classify the same set independently and we can compare results from each.

@author: Peter
"""

import numpy
import math
import os
import csv
import time
from random import shuffle
from PIL import Image, ImageDraw, ImageFont


image_directory = "Images/Cores/"
map_directory = "Binaries/Output_Maps/"
csv_directory = "Results/Validation/"

segment_dimension = 12
step_size = 6


def verify(num_classes=13, num_examples_per_class=5): # 13*5 = 65
    
    # ask user to select which dataset to perform validation on
    target_csv = pick_target_file()
    # either generate a new set of examples or load a set which was used previously
    if target_csv is None:
        # ask for user name - this will be appended to csv filenames to make them easier to identify
        username = input("Please enter your name: ")
        save_name = "validation_examples_" + username + time.strftime('_%a_%I_%M_%p')
        examples = generate_examples(num_classes, num_examples_per_class)
        # shuffle examples to ensure that they are presented to the user in a random fashion (want the user to treat each example independently)
        shuffle(examples)
        save_examples(examples, save_name)
    else:
        examples = load_examples(target_csv)
        save_name = target_csv
        # if loading data then we don't perform a shuffle as this would make it difficult to compare results (to do this with shuffling you would need to index)
    comparisons = []
    ambiguous = 0
    # iterate through list of machine classified examples asking for user input
    for e in examples:
        # show example to user
        im = fetch_image(e)
        im.show()
        # ask user to input classification and resolve appropriately
        possible_classes = list(range(num_classes))
        valid_entry = False
        while not valid_entry:
            try: user_label = int(input("Enter category number: "))
            except ValueError:
                print("Please enter a valid integer value. Enter '-1' if you want to open a new instance of the image")
                continue
            if user_label == -1: im.show()
            elif user_label == 255:
                # generate new point and record that an ambiguous point was found
                ambiguous += 1
                e = get_example(e[0])
                im = fetch_image(e)
            elif user_label in possible_classes:
                comparisons.append((e[0], user_label)) # tuple: (machine_guess, user_input)
                valid_entry = True
            # if no appropriate input was found then valid_entry=False so we go back to the start of the while loop and ask user for another input
        im.close()
    
    # once we have all the user inputs we want to compare them with the original labels
    print(comparisons)
    save_guesses(comparisons, save_name)
    # produce stats and a confusion matrix
    
### end Output_Valudation.verify

def pick_target_file(prefix="validation_examples"):
    
    # use while loop to control user input - creates an infinate loop which only ends if they enter a valid response
    while True:
        new_or_old = input('Would you like to create a new set of validation images or use a previously generated set of examples?\nType "0" to generate a new set\nType "1" to use a previous set\n')
        if new_or_old == "0": return(None)
        elif new_or_old == "1": break
        else: print("Invalid entry")
    file_list = os.listdir(csv_directory)
    file_list.sort()
    index = {}
    count = 0
    print("Files available to load:")
    for f in file_list:
        if f.startswith(prefix):
            print(count, f)
            index[count] = f
            count += 1
    if count == 0:
        print("No previous examples available")
        return(None)
    while True:
        selection = input('Please enter a number corresponding to the appropriate file. Alternatively type "n" to generate a new set instead.\n')
        if selection == "n": return(None)
        elif int(selection) in index: return(index[int(selection)])
        else: print("Invalid entry")
    
### end Output_Validation.pick_target_file

def generate_examples(num_classes, num_examples_per_class):
    
    print("Generating new list of examples for {} classes with {} examples per class".format(num_classes, num_examples_per_class))
    # generate list of labels
    total_examples = num_classes * num_examples_per_class
    labels = [math.floor((i/total_examples)*num_classes) for i in range(total_examples)]
    # we want to find an example for each of these labels (i.e. 5 examples of laumontite, 5 examples of epidote, etc)
    
    # create a list of examples (piture locations) corresponding with each label
    examples = []
    for label in labels:
        examples.append(get_example(label))
        
    return(examples)
    
### end Output_Validation.generate_examples

def get_example(label):
    
    # seach through each of the map files to find an example corresponding to the label
    # this method is heavily biased towards finding results from files containing a small amount of a particular class
    # this could be particularly influential on your results for rare classes
    # e.g. if your classifier correctly identifies 100 pixels in one core but then misclassifies 1 pixel in three other cores then this method will very frequently pick the misclassified pixels despite them being relatively uncommon
    # consider re-writing to fully randomise example selection
    
    map_list = os.listdir(map_directory)
    shuffle(map_list)
    for f in map_list:
        if f.endswith("meta.npy"): continue  # skip the meta files (we don't want to find any images there!)
        stack= numpy.load(map_directory + f)
        if label in stack:
            index = list(numpy.ndindex(stack.shape))
            shuffle(index)
            for i in index:
                if stack[i] == label:
                    location = i
                    break
            print("Example: class {}".format(stack[location]))
            
            x_coord = location[1] * step_size
            y_coord = location[0] * step_size
            filename = f[:-4] # return the filename without the ".npy" extension
            break # end loop
        else: continue
    
    return((label, filename, x_coord, y_coord)) # return tuple
    
### end Output_Validation.get_example

def fetch_image(example, display_size=500):
    
    # deconstruct tuple to get values
    image_name, x, y = example[1], example[2], example[3]
    # open image
    im = Image.open(image_directory + image_name + ".jpg")
    
    # draw a square around the 12x12 segment (so the user can see what they are being asked to classify)
    draw = ImageDraw.Draw(im)
    #    draw.line(startx, starty, endx, endy)
    
    # assume x,y is the top left coordinate of the 12x12 box
    draw.line((x-2, y-2, x+12, y-2), fill=(0,255,0), width=2) # top
    draw.line((x-2, y-2, x-2, y+12), fill=(0,255,0), width=2) # left
    draw.line((x+12, y-2, x+12, y+12), fill=(0,255,0), width=2) # right
    draw.line((x-2, y+12, x+13, y+12), fill=(0,255,0), width=2) # bottom
    del draw
    
    # crop image to a small/medium sized area (helps to highlight the area we are interested in while preserving context)
    # first adjust x and y to ensure that the user is presented with a full 200x200 area rather than one clipped by image dimensions
    lower = int((display_size / 2) - (segment_dimension / 2))
    upper = int((display_size / 2) + (segment_dimension / 2))
    
    x = max(lower, x)
    x = min(x, im.size[0]-upper)
    y = max(lower, y)
    y = min(y, im.size[1]-upper)
    
    # then crop
    box = (x-lower, y-lower, x+upper, y+upper)    # notice that x, y represent the top left corner of our 12x12 segment. Hence the crop is offset to keep the segment central.
    im = im.crop(box)
    
    return(im)
    
### end Output_Validation.fetch_image

def load_examples(filename):
    
    examples = []
    with open(csv_directory + filename, newline='') as csvfile:
        # setup reader function
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # skip headings
        next(datareader)
        # iterate through the file to find all examples (one example per row)
        while True:
            try:
                row = next(datareader)
                examples.append((int(row[0]), row[1], int(row[2]), int(row[3])))
            except StopIteration:
                break
    print("Examples loaded from: " + filename + ".csv")
    
    return(examples)
    
### end Output_Validation.load_examples

def save_examples(examples, save_name):
    
    with open(csv_directory + save_name + ".csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # write headings
        datawriter.writerow(["Category", "Core", "x-coord", "y-coord"])
        # write each example to the file with one example per row
        for e in examples:
            datawriter.writerow(e)
    # unindent to close file, declare success        
    print("Examples saved in " + csv_directory + save_name + ".csv")
    
### end Output_Validation.save_examples

def save_guesses(comparisons, examples_save_name):
    
    comparison_save_name = "comparisons_" + examples_save_name[20:25] + time.strftime('_%a_%I_%M_%p')
    with open(csv_directory + comparison_save_name + ".csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # write name of corresponding file containing the example data
        datawriter.writerow([examples_save_name + ".csv"])
        # write headings
        datawriter.writerow(["Machine_guess", "Human_guess"])
        # write data
        for c in comparisons:
            datawriter.writerow(c)
    print("Comparisons saved as " + csv_directory + comparison_save_name + ".csv")
    
### end Output_Validation.save_guesses

def display_errors(num_cols=5, num_rows=2):
    
    num_errors_to_display= num_cols * num_rows
    # start by selecting a verification run which contains some errors
    filename = pick_target_file(prefix="comparisons_")
    with open(csv_directory + filename, newline='') as csvfile:
        # setup reader function
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # get the correct filename containing corresponding example information
        examples_file = next(datareader)
        examples_file = examples_file[0]
        # skip headings
        next(datareader)
        # iterate through the file to find all eroneous guesses and their corresponding location within the list of examples
        errors = {}
        example_num = 0
        while True:
            try:
                row = next(datareader)
                if int(row[0]) != int(row[1]): errors[example_num] = int(row[1]) # if human guess doesn't match the machine guess then record the example number and the human guess
                example_num += 1
            except StopIteration:
                break
    # open filename containing the example information and attach this to each error
    with open(csv_directory + examples_file, newline='') as csvfile:
        # setup reader function
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # skip headings
        next(datareader)
        # get example informtaion for each error and add it to examples list (notice that this is formatted slightly differently to the examples lists from elsewhere in the module - this doesn't mater)
        examples = []        
        example_num = 0
        while True:
            try:
                row = next(datareader)
                if example_num in errors: examples.append((int(row[0]), row[1], int(row[2]), int(row[3]), errors[example_num]))
                example_num += 1
            except StopIteration:
                break
    # shuffle examples and trim to desired length
    shuffle(examples)
    examples = examples[0:num_errors_to_display]
    # iterate through list of examples to create display of non-matching guesses
    display = Image.new("RGB", (201*num_cols, 250*num_rows), color=(0,0,0))
    for j in range(num_rows):
        for i in range(num_cols):
            display_num = i + (j * 5)
            try:
                current = examples[display_num] # fetch the current example - if we have run out this will produce IndexError
                text = "Human guess: {} \nMachine guess: {}".format(current[4], current[0])
                im = fetch_image(current, display_size=200)
                textbox = Image.new("RGB", (200, 50), color=(255,255,255))
                # get a font and write text into textbox
                ## use a truetype font
                font = ImageFont.truetype("arial.ttf", 20)
                draw_text = ImageDraw.Draw(textbox)
                draw_text.text((5, 5), text, font=font, fill=(0,0,0))
                # paste image and textbox into appropriate place in display
                display.paste(im, (i*201,j*251))
                display.paste(textbox, (i*201,(j*251)+200))
            except IndexError:
                break # the remaining space remains blank
    display.show()
        
### end Output_Validation.display_errors