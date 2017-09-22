# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:48:09 2017

Takes category map and attempts to remove anomolies.

Beware that it does this in a relatively unsophisticated way. Need to set variables according to map resolution.

@author: Peter
"""

def reduce_noise(file_name, filter_strength=3):
# filter strength should probably be an integer between zero and ten, defaults to the sensible value of 3
    
    import numpy
    
    # set directories to find/save
    target_folder = "Binaries/"
    destination_folder = "Binaries/"
    
    # unpack data   
    load_data = numpy.load(target_folder + file_name + ".npy")
    stack = load_data.tolist()
    load_metadata = numpy.load(target_folder + file_name + "_meta" + ".npy")
    metadata = load_metadata.tolist()

    map_width = metadata[0]
    map_height = metadata[1]
    count = 0
    
    # iterate through image to check for anomalies, if detected merge these pixels into the local background
    for j in range(map_height):
        for i in range(map_width):
            neighbourhood = check_local_area(stack, map_width, map_height, i, j)
            # if there are lots of background pixels in the neighbourhood then assign this one as background/fault
            # notice that this means the noise reduction is much more aggressive in background/fault zones
            if neighbourhood[1] > 50:
                stack[j][i] = 10
                count += 1
            # if there are few similar pixels nearby then assume that this is an erroneous result and overwrite it based on nearby pixels
            elif neighbourhood[0] < filter_strength:
                stack[j][i] = return_local_area(stack, map_width, map_height, i, j)          
                count += 1
                
    """# do a second pass, this time starting bottom right
    for j in range(map_height-1, -1 ,-1):
        for i in range(map_width-1, -1, -1):
            neighbourhood = check_local_area(stack, map_width, map_height, i, j)
            if neighbourhood[1] > 55:
                stack[j][i] = 10
                count += 1
            elif neighbourhood[0] < 3:
                stack[j][i] = return_local_area(stack, map_width, map_height, i, j)          
                count += 1"""
                
    # re-pack and save
    output = numpy.asarray(stack)
    numpy.save(destination_folder + file_name + ".npy", output)
    
   # create a new metafile to accompany the map
    numpy.asarray(metadata)
    numpy.save(destination_folder + file_name + "_meta" + ".npy", metadata)
    
    # proclaim success
    print("Noise reduction complete, new map saved as: " + file_name + ".npy") 
    print("{} pixels removed as noise".format(count))
    print("This represents {}% of the total surface".format(count*100/(map_height*map_width)))

### end Map_Smoother.reduce_noise

def check_local_area(stack, map_width, map_height, x_coord, y_coord):
# inspect 9x9 grid containing the current pixel
    
    neighbours = -1     # pixel will always be neighbours with itself
    background = 0
    local_pixel = stack[y_coord][x_coord]
    for y in range(y_coord-4, y_coord+4, 1):
        for x in range(x_coord-4, x_coord+4, 1):
            if -1 < x < map_width and -1 < y < map_height:  # check that the pixel is referencable
                if stack[y][x] is local_pixel:
                    neighbours += 1
                if local_pixel != 10 and stack[y][x] is 10:
                    background += 1
            else: background += 1
                
    return[neighbours, background]

### end Map_Smoother.check_local_area

def return_local_area(stack, map_width, map_height, x_coord, y_coord):
# inspect 5x5 grid and return the most common element

    zone_elements = []
    for y in range(y_coord-2, y_coord+2, 1):
        for x in range(x_coord-2, x_coord+2, 1):
            if -1 < x < map_width and -1 < y < map_height:  # check that the pixel is referencable
                zone_elements.append(stack[y][x])
    commonest_element =  max(zone_elements, key=zone_elements.count)
    
    return(commonest_element)

### end Map_Smoother.return_local_area