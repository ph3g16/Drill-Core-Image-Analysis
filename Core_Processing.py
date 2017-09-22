# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:31:52 2017

PTOTOTYPE CODE - not used in final model, this older version of the code shows how Human_Map and Map_Blender.overlay were intended to be implemented

Control module. Calls other modules in order to process data from a single core.

Prior to use the core image should be rolled (if possible) and a boundary file should be created.

@author: Peter
"""

#import modules in roughly the order of use
import Alpha_Validation
import Raw_Image_Processor
import Cylindrical_Image_Prep
import Map_Generator
import Human_Map
import Map_Blender
import Map_Smoother
import Plane_Analysis
import Map_Analysis

# should output a series of pictures showing the curves plus a CSV file

core_name = "CS_blah_blah"
### or.... for core_name in [list]

# check that this bit of rock doesn't have any lime green in it
Alpha_Validation.validate(core_name)

# package image as a binary which can be understood by the machine classifier
Raw_Image_Processor.extract(core_name)

# trim some data from the original image to ensure that it corresponds exactly with the map
Cylindrical_Image_Prep.trim(core_name, save=True, receive_mode=".jpg", delete_old=True)

# also trim boundary input picture
Cylindrical_Image_Prep.trim("../Bounds/" + core_name, save=True, receive_mode=".png", delete_old=False)

# use machine classifier to generate a classification map
Map_Generator.classify(core_name)  # notice that in this case core_name is actually being used as a save name rather than to find the correct file

# transform the defined boundary areas into a map file
Human_Map(core_name)

# overlay the boundary areas onto the classification map
Map_Blender.overlay(core_name + "_bounds", core_name, core_name)

# reduce noise
Map_Smoother.reduce_noise(core_name)

# analyse the finished map to extract curve information
Plane_Analysis.find_all_curves(core_name + "_denoised")

# analyse the map for other useful information
Map_Analysis.quantify(core_name + "_denoised", by_line=False)