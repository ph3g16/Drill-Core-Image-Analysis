# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:27:46 2017

Module which iterates through binary maps to create quantity and/or plane information files.

You probably don't need this - it is a variation of Core_Processing_Alt.py

@author: Peter
"""

import os
import Map_Analysis
import Plane_Analysis

def process_multiple():
    
    file_list = os.listdir("Binaries/Output_Maps")
    file_list.sort()
    for core in file_list:
        core_name = core[:-4] # remove the last four characters from the filename (.jpg)
        if not core_name.endswith("meta"):
            core_name = core[:-4] # remove the last four characters from the filename (.jpg)
            print(core_name)
            Map_Analysis.quantify(core_name, by_line=True)

### end Core_Processing.process_multiple

