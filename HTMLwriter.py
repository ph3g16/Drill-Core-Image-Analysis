# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:44:02 2017

Script to generate html description of borehole.

Writing the file location and absolute position of every image would be dull!

@author: Peter
"""

import os

core_folder = "Images/Cores/"
map_folder = "Images/Maps/"
save_destination = "HTML/"

def write_webpage():
    
    # save text as a .txt file (manually edit to .html to create a webpage)
    with open(save_destination + "webpage" + ".html", "w+") as text:
        text.write('<!DOCTYPE html>\r\n')
        text.write('<html>\r\n')
        text.write('<head>\r\n')
        text.write('<style>\r\n')
        text.write('</style>\r\n')
        text.write('</head>\r\n')
        text.write('<body>\r\n')
    
        # add headings and style information
        
        # add scale
        
        # read csv to get depth of each core
        
        
        count = 0
        # read core_folder to get all JPEG files and add them to the webpage
        core_list = os.listdir(core_folder)
        core_list.sort()
        for f in core_list:
            src = "file:///C:/Oman%20Drilling%20Project/Final%20Model/" + core_folder + f
            alt = f
            top = int(100 + count)
            left = 20
            text.write('<img src="{}" alt="{}" syle="position:absolute;top={}px;left={}px;">\r\n'.format(src, alt, top, left))
            
        # read map folder to get all PNG maps and add them to the webpage
        map_list = os.listdir(map_folder)
        map_list.sort()
        for f in map_list:
            src = "file:///C:/Oman%20Drilling%20Project/Final%20Model/" + map_folder + f
            alt = f
            top = int(100 + count)
            left = 2000
            text.write('<img src="{}" alt="{}" syle="position:absolute;top={}px;left={}px;">\r\n'.format(src, alt, top, left))
            
        # closing statements
        text.write('<body>\r\n')
        text.write('<body>\r\n')
        
    # close .txt file
    
### end HTMLwriter.write_webpage