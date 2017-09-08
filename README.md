# Drill-Core-Image-Analysis
Uni Soton research project

Program requires installation of tensorflow and PIL packages in whichever python environment you are using. Code was written for python 3.5, tensorflow version 1.3 and was run from an anaconda environment. Other setup configurations should work but have not been tested.

Where possible I have tried to make the program modular with a philosophy of one module per process/idea. The vast majority of the time you will only want to use a few of the modules (documented under "Key operations"). The remaining modules have typically been written with a single purpose in mind such as examining the output data to find a sine curve.

All modules are commented and have a description text. The vast majority are intended to be imported into a python console and run by calling a function from there. If you are unsure what a module does the best way to find out is to read the description text and try to run whatever looks like the most important function from inside python.

#### Key operations

To train the classifier:
 - In python enter `import Extract_Handler as EH` then `EH.extract()`
 - Quit python and enter `python cifar10_train.py` into the command terminal
This will start training and create a new checkpoint save file. It will also delete the current checkpoint so you might want to copy and paste the "mode2_training" folder before training a new version (I usually archive useful checkpoints by copying them into the "Good Checkpoints" folder, which has no other function).

To process one of the core files and turn it into a prediction map:
 - In python enter `import Raw_Image_Processor as RIP` then `import Map_Generator as Mg` then `import Map_Viewer as Mv`
 - In python enter `RIP.extract("CS_5057_2_A_055_3_1", 12, 6)` which creates a set of temporary binary files (these are labelled "raw_batch_x" and appear in `Binaries/`)
 - In python enter `Mg.classify("test", 2)` which will generate a prediction map and store it as a numpy array, "test" is the save name.
 - In python enter `Mv.view("test")` to tanslate the numpy array into a PNG image. This image will be saved in `Images/Maps/` but it will also pop up in a new window for you to view immediately.
 
To process lots of cores all in one go:
 - In python enter `import Core_Processing_Alt as CPA`
 - In python enter `CPA.process_multiple`
This is set up to process ALL the cores which will take a very long time. However, you can also use it to process a subset by amending the "prefix" variable within Core_Processing_Alt.py

#### Other operations

To create a training set and a seperate evaluation set:
 - In python enter `import Extract_Handler as EH` then `EH.extract(1)`. Notice the crucial argument "1" which forces the extract handler to partition off some of the training data for use as evaluation data.
 - Quit python and enter `python cifar10_train.py` to train the classifier.
 - enter `python cifar10_eval.py` to run an evaluation function.
 - a confusion matrix will be generated and saved as a csv file in `Results/Validation/` (open this in excel)
 
To perform cross validation:
 - In python enter `import Extract_Handler as EH` then `EH.extract(2)`. Notice the crucial argument "2" which forces the extract handler to sort the data into x many independent sets.
 - To ensure independence you should go into the code for Extract_Handler.py and change the "step_size and "segment_dimension" variables to be equal. Otherwise you will have overlap between images and hence the sets will not be fully independent.
 - In python enter `import Cross_Validation as Cv` then `Cv.get_stats()`. This will run the training routine multiple times so can take a while to complete.
 
To train the classifier using a different training set:
 - Open `Images/Training2` in file explorer
 - This is the folder that training images are taken from (Extract_Handler.py is used to process the training images). Drag and drop images into this file to add them to the training data. Delete files (or put them into one of the holding folders) to remove them from the training data.
 - Extract_Handler.py will assign a class to your images automatically depending on the first 4 letters of each image file. If you are using training images which have different prefixes to the ones I have used you will need to edit Extract_Handler.py to add your prefixes (look in the "get_category()" function).
 - train the classifier (as described above)
 
#### Suggestions for modifying or reusing the code
 
The classifier modules (cifar10, cifar10_train, cifar10_eval, cifar10_input) were built by modifying a tensorflow tutorial: https://www.tensorflow.org/tutorials/deep_cnn
They have been substantially modified and as a result the code is a bit frankenstein-ish with a couple of irritating quirks. If you want to use the code then it works as is but if you anticipate making changes to the input/output structure then you would be well advised to completely rewrite these modules (from scratch) although there are some ideas and code snippets which you might want to reuse such as the graph strucute and confusion matrix evaluation output.

The other modules are more linear and should be easier to understand and amend. The main quirk is the way in which information is passed between the modules:
 - Extract_Handler and Raw_Image_Processor both use Image_Extractor to read images and convert them into a cifar10 style binary format (google cifar10 - this is the formatting that the classifier modules are built to read)
 - Map_Generator produces data which is stored as an array using numpy.save
 - Almost all the other modules are used to process data created using Map_Generator. They load the data using numpy.load and resave it (if applicable) using numpy.save.
 - Although numpy is used for most load/save operations the array will typically be pushed into a list format where each row is a list and the rows are contained within another list. So, if you wanted to access a prediction with coordinate x,y you would type listname[y][x] to access that specific element.
 - Map_Analysis, Plane_Analysis and cifar10_eval all save information in a CSV format. This is because you might want to manipulate the data in excel or use it to generate graphs or similar. You can also load this data back into python using a CSV reader (Map_Analsysis.py and Cross_Validation.py contain code that does this).
