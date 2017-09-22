# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:31:52 2017

Control module. Calls other modules in order to process data from a single core.

Prior to use the core image should be rolled (if possible) and a boundary file should be created.

Core_Processing_Alt differs from Core_Processing because it 

@author: Peter
"""

#import modules in roughly the order of use
import Alpha_Validation
import Raw_Image_Processor
import Map_Generator
import Map_Smoother
import Map_Viewer
import Plane_Analysis
import Map_Analysis
import math
import time
import os

def process_multiple():
    
    prefix = "CS_5057_2_A_" # variable used for alternate if statement below
    validations = []
    
    file_list = os.listdir("Images/Cores")
    file_list.sort()
    already_processed = os.listdir("Images/Maps")
    already_processed = [file[:-4] for file in already_processed] # remove the ".png" attachment
    
    for core in file_list:
        core = core[:-4] # remove the last four characters from the filename (.jpg)
#        if core.startswith(prefix):
        if core not in already_processed:    
            print(core)
            validations.append( process_core(core_name=core) )
            time.sleep(10) # pause for 20 seconds
    
    print(validations)

### end Core_Processing.process_multiple

# should output a series of pictures showing the curves plus a CSV file
def process_core(core_name):
    ### or.... for core_name in [list]
    
    # start the clock
    start_time = time.time()
    
    # check that this bit of rock doesn't have any lime green in it
    validate = Alpha_Validation.validate(core_name)
    
    # convert core image to binary and generate corresponding map file
    Raw_Image_Processor.extract(core_name, 12, 6)
    Map_Generator.classify(core_name)
    Map_Smoother.reduce_noise("Output_Maps/" + core_name, 4)
    
    # create a png representation of the map but 
    Map_Viewer.view(core_name, show_im=False)
    
    # analyse the finished map to extract curve information
#    Plane_Analysis.find_all_curves(core_name)
    
    # analyse the map for other useful information
    Map_Analysis.quantify(core_name, by_line=True)
    
    # end the clock
    end_time = time.time()
    
    # calculate and print total time for cycle
    runtime = end_time - start_time # runtime is measured in seconds
    minutes = math.floor(runtime / 60)
    seconds = runtime - 60 * minutes
    print("Runtime: {} minutes and {} seconds".format(minutes, seconds))
    
    return(validate)

### end Core_Processing.process_core