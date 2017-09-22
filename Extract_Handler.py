# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:43:07 2017

Use this to pass arguments to Image_Extractor.py

This is adapted from previous versions and selects files based on prefixes rather than requiring each filename to be entered.
If targetting new/different training data prefixes you need to amend the "get_category" function.
Would be easy to redesign to drag + drop folder structure with a folder for each category (useful if you want to distribute to non-programmers)

Alternate save options have been added to make it easier to build evaluation sets and perform cross-validation.

Be careful if editing the code which performs alternate save modes. In order to fully randomise batches while working
around memory limit the program saves temporary holding files which it then loads at a later stage. The code which
operates the save/load between temporary states is different to the code which saves batches in their final state ready
to be used as classifier inputs.

@author: Peter
"""

# hard-coded variables

target_directory = "Binaries/"
image_directory = "Images/Training2/"

alpha_channel = (0,255,1)   # I use 00ff00 as my filter colour - this translates as 0,255,1 when exported from GIMP to JPEG

segment_dimension = 12
step_size = 3

record_length = segment_dimension**2 * 3 + 1  # number of list elements per 12x12 picture extracted

limit = 4000 # discard any images from a training file which exceed this hard limit

HSV = False # if true converts all images to HSV mode rather than RGB

num_categories = 20 # arbitrary but must be higher than largest category label

# imports

import Image_Extractor
import numpy
import math
import os
from random import shuffle

# functions ("extract" is the key function which should be called from python)

def extract(save_method=0):
    
    # delete existing training/eval/cross batch binaries
    for f in os.listdir(target_directory):
            if f.startswith("train_batch"): os.remove(target_directory + f)
            if f.startswith("eval_batch"): os.remove(target_directory + f)
            if f.startswith("cross_batch"): os.remove(target_directory + f)
            
    # delete any temporary binaries (could exist if the code didn't complete on the last runthrough)
    for f in os.listdir(target_directory):
            if f.startswith("hold"):
                os.remove(target_directory + f)
         
    # initialise variables
    save_count = 0
    total_records = 0
    stack = []
    category_count = [0 for i in range(num_categories)]
    
    # generate shuffled file list
    file_list = os.listdir(image_directory)
    shuffle(file_list)   
    # iterate through files within selected folder
    for file_name in file_list:
        category = get_category(file_name)
        if category is None: return
        if category is False: continue
        hold, record_count = Image_Extractor.harvest(image_directory + file_name, category, alpha_channel, segment_dimension, step_size, limit, HSV=HSV)
        category_count[category] += record_count
        stack.extend(hold)
        hold = None
        if len(stack) > 14000000: # effectively a memory limit - if stack exceeds 20MB then save and start writing a new data file
            try: save(stack, save_count, save_method)
            except ValueError: print(ValueError)
            total_records += int(len(stack)/record_length)
            save_count += 1
            stack = []  
    # once we have iterated through all the files we want to save data to binary
    try: save(stack, save_count, save_method)
    except ValueError: print(ValueError)
    total_records += int(len(stack)/record_length)
    save_count += 1
    
    # perform reshuffling routine (if applicable)
    if save_method is 1: save_train_and_eval_sets(num_hold_files=save_count, num_train_files=8, num_eval_files=2, total_records=total_records)
    if save_method is 2: save_cross_validation_set(num_hold_files=save_count, num_cross_files=5, total_records=total_records)

    print(category_count)

### end ExtractHandler.extract

def get_category(file_name):
    
    prefix = file_name[0:4]
    if prefix == "Edge":
        return(0)
    if prefix == "Faul":
        return(0)
    if prefix == "Mark":
        return(0)
    elif prefix == "Rock":
        return(1)
    elif prefix == "Othe": # the "other" prefix is used for testing the viability of issolating a specific feature
        return(1)
    elif prefix == "Red_":
        return(2)
    elif prefix == "RedC":
        return(2)
    elif prefix == "Blue":
        return(3)
    elif prefix == "Yell":
        return(4)
    elif prefix == "Laum":
        return(5)
    elif prefix == "Alb_":
        return(6)
    elif prefix == "Spec":
        return(6)
    elif prefix == "Oliv":
        return(7)
    elif prefix == "AltO":
        return(7)
    elif prefix == "Epi_":
        return(8)
    elif prefix == "Epid":
        return(8)
    elif prefix == "Beig":
        return(8)
    elif prefix == "Alte":
        return(9)
    elif prefix == "Oxid":
        return(10)
    elif prefix == "Plag":
        return(11)
    elif prefix == "Clin":
        return(12)
    elif prefix == "Anhy":
        return(5) # 13
    elif prefix == "Pren":
        return(14)
    elif prefix == "Quar":
        return(5) # 15
    elif prefix == "Hold":
        return(False) # skips the holding folder
    else:
        print("Unknown prefix: {}".format(prefix))
        return(None)
    
### end Extract_Handler.get_category

def save(stack, save_count, save_method):
    
    if save_method is 0:
        basic_save(stack, save_count)
    elif save_method is 1:
        save_holding_file(stack, save_count)
    elif save_method is 2:
        save_holding_file(stack, save_count)
    else: raise ValueError("Save method invalid")

def basic_save(stack, save_count):
    
    # save the extracted data as a training set
    save_name = "train_batch_{}".format(save_count)
    output = numpy.array(stack, numpy.uint8)
    output.tofile(target_directory + save_name + ".bin")
    print("Extract successful, binary file is: " + save_name + ".bin")
    
### end Extract_Handler.basic_save

def save_holding_file(stack, save_count):
    # save data into a temporary holding file - this will be accessed randomly at a later stage to create the training/eval batches
    # this two stage solution is required due to memory limits.
    
    save_name = "hold_{}".format(save_count)
    output = numpy.array(stack, numpy.uint8)
    numpy.save(target_directory + save_name + ".npy", output)
    print("Extract successful, data put into binary holding file: " + save_name + ".npy")
    
### end Extract_Handler.save_holding_file

def save_train_and_eval_sets(num_hold_files, num_train_files, num_eval_files, total_records):
    
    # generate empty files ready to append data
    for train_num in range(num_train_files):
        batch = numpy.array([]) # save empty list as a numpy array
        save_name = "train_batch_{}".format(train_num)
        numpy.save(target_directory + save_name + ".npy", batch)
        
    for eval_num in range(num_eval_files):
        batch = numpy.array([]) # save empty list as a numpy array
        save_name = "eval_batch_{}".format(eval_num)
        numpy.save(target_directory + save_name + ".npy", batch)
    
    # set total_records to an exact multiple of the number of target files
    # this stage is necessary if you want the files to contain exacty the same number of records (otherwise there is likely to be a discrepancy of 1 record between the batch sizes)
    remainder = total_records % (num_train_files + num_eval_files)
    total_records = total_records - remainder
    
    # create list declaring where each piece of data will be sent
    target = [math.floor((i/total_records)*(num_train_files + num_eval_files)) for i in range(total_records)]
    for i in range(remainder): target.append(None) # extends list of targets to meet actual number of records - any records with a "None" target will get skipped later on
    shuffle(target) # shuffle our list of numbers which we already defined to have a known range and uniform distribution
    bookmark = 0
    
    # iterate through each temporary file and push contents to the appropriate training or eval batch
    for x in range(num_hold_files):
        
        stack = numpy.load(target_directory + "hold_{}".format(x) +".npy")
        stack = stack.tolist()
        
        # redefine the stack to store each record as an independent list (enables easier shuffling and sorting)
        stack = [stack[(i*record_length): (i* record_length)+record_length] for i in range(int(len(stack)/record_length))]
        
        num_records = len(stack)
        # attach destination information to each record
        for i in range(num_records):
            stack[i].insert(0, target[i+bookmark])
        bookmark = bookmark + num_records
        
        for train_num in range(num_train_files):
            # open file
            save_name = "train_batch_{}".format(train_num)
            batch = numpy.load(target_directory + save_name + ".npy")
            batch = batch.tolist()
            # append relevant results to the file
            for element in stack:
                if element[0] == train_num: batch.append(element[1:])
            # save/close the file
            batch = numpy.array(batch)
            numpy.save(target_directory + save_name + ".npy", batch)
        
        for eval_num in range(num_eval_files):
            # open file
            save_name = "eval_batch_{}".format(eval_num)
            batch = numpy.load(target_directory + save_name + ".npy")
            batch = batch.tolist()
            # append relevant results to the file
            for element in stack:
                if element[0] == (eval_num + num_train_files): batch.append(element[1:])
            # save/close the file
            batch = numpy.array(batch)
            numpy.save(target_directory + save_name + ".npy", batch)
    
    # finally, open each batch and shuffle - to remove bias in order created by sampling process
    # also this time we save it using the "tofile" method which puts it in a format which can be read by the classifier
    for train_num in range(num_train_files):
        save_name = "train_batch_{}".format(train_num)
        batch = numpy.load(target_directory + save_name + ".npy")
        batch = batch.tolist()
        shuffle(batch)
        # need to change formatting from list of records back to an unbroken list of integers
        stack = []
        for element in batch:
            stack.extend(element)
        batch = numpy.asarray(stack, numpy.uint8)
        batch.tofile(target_directory + save_name + ".bin")
        print(save_name + ".bin saved")
        
    for eval_num in range(num_eval_files):
        save_name = "eval_batch_{}".format(eval_num)
        batch = numpy.load(target_directory + save_name + ".npy")
        batch = batch.tolist()
        shuffle(batch)
        # need to change formatting from list of records back to an unbroken list of integers
        stack = []
        for element in batch:
            stack.extend(element)
        batch = numpy.asarray(stack, numpy.uint8)
        batch.tofile(target_directory + save_name + ".bin")
        print(save_name + ".bin saved")
    
    # iteratively delete temporary holding files
    for f in os.listdir(target_directory):
            if f.endswith(".npy"):
                os.remove(target_directory + f)
                
    print("Temp file cleanup completed")

### end Extract_Handler.save_train_and_eval_sets

def save_cross_validation_set(num_hold_files, num_cross_files, total_records):
    
    # generate empty files ready to append data
    for cross_num in range(num_cross_files):
        batch = numpy.array([]) # save empty list as a numpy array
        save_name = "cross_batch_{}".format(cross_num)
        numpy.save(target_directory + save_name + ".npy", batch)
    
    # set total_records to an exact multiple of the number of target files
    # this stage is necessary if you want the files to contain exacty the same number of records (otherwise there is likely to be a discrepancy of 1 record between the batch sizes)
    remainder = total_records % (num_cross_files)
    total_records = total_records - remainder
    
    # create list declaring where each piece of data will be sent
    target = [math.floor((i/total_records)*(num_cross_files)) for i in range(total_records)]
    for i in range(remainder): target.append(None) # extends list of targets to meet actual number of records - any records with a "None" target will get skipped later on
    shuffle(target) # shuffle our list of numbers which we already defined to have a known range and uniform distribution
    bookmark = 0

    # iterate through each temporary file and push contents to the appropriate batch
    for x in range(num_hold_files):
        
        stack = numpy.load(target_directory + "hold_{}".format(x) +".npy")
        stack = stack.tolist()
        
        # redefine the stack to store each record as an independent list (enables easier shuffling and sorting)
        stack = [stack[(i*record_length): (i* record_length)+record_length] for i in range(int(len(stack)/record_length))]
        
        num_records = len(stack)
        # attach destination information to each record
        for i in range(num_records):
            stack[i].insert(0, target[i+bookmark])
        bookmark = bookmark + num_records
        
        for cross_num in range(num_cross_files):
            # open file
            save_name = "cross_batch_{}".format(cross_num)
            batch = numpy.load(target_directory + save_name + ".npy")
            batch = batch.tolist()
            # append relevant results to the file
            for element in stack:
                if element[0] == cross_num: batch.append(element[1:])
            # save/close the file
            batch = numpy.array(batch)
            numpy.save(target_directory + save_name + ".npy", batch)
    
    # finally, open each batch and shuffle - to remove bias in order created by sampling process
    # also this time we save it using the "tofile" method which puts it in a format which can be read by the classifier
    for cross_num in range(num_cross_files):
        save_name = "cross_batch_{}".format(cross_num)
        batch = numpy.load(target_directory + save_name + ".npy")
        batch = batch.tolist()
        shuffle(batch)
        # need to change formatting from list of records back to an unbroken list of integers
        stack = []
        for element in batch:
            stack.extend(element)
        batch = numpy.asarray(stack, numpy.uint8)
        numpy.save(target_directory + save_name + ".npy", batch)
        print(save_name + ".npy saved")
    
    # iteratively delete temporary holding files
    for f in os.listdir(target_directory):
            if f.startswith("hold_"):
                os.remove(target_directory + f)

    print("Temp file cleanup completed")
    
### end Extract_Handler.save_cross_validation_set

