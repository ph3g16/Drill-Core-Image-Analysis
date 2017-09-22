# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:05:14 2017

Module to perform cross validation.

Create binary fines using Extract_Handler (save_mode=2) before running this code.
The module will iteratively open and resave/rename them before performing each training/eval run.

Cross validation involves multiple training runs (usually 5-10) so obviously will take a significant amount of time to complete.

@author: Peter
"""

import numpy
import os
import csv
import cifar10_train
import cifar10_eval
from time import sleep

num_files = 5

target_directory = "Binaries/"
csv_directory = "Results/Validation/"

def get_stats():
# main method
    """
    # iteratively delete any confusion matrixes stored in csv_directory
    for f in os.listdir(csv_directory):
        if f.startswith("confusion_matrix"): os.remove(csv_directory + f)

    # count the number of files to cross validate
    num_files = 0
    for f in os.listdir(target_directory):
        if f.startswith("cross_batch"): num_files += 1
    
    # iterate through the files assigning a different eval set each time (remaining files are used as training set)
    for x in range(num_files):
        
        # create training data
        for i in range(num_files):
            # skip file x - we don't want to include this in the training data
            if i is x: continue
            # open
            stack = numpy.load(target_directory + "cross_batch_{}".format(i) + ".npy")
            # save
            stack.tofile(target_directory + "train_batch_{}".format(i) + ".bin")
        # create eval data - open and save set x as the eval set
        stack = numpy.load(target_directory + "cross_batch_{}".format(x) + ".npy")
        stack.tofile(target_directory + "eval_batch_{}".format(x) + ".bin")

        # run training
        cifar10_train.main()
        
        # run eval
        # the eval step will print results on screen but also produces a CSV file containing the results
        cifar10_eval.main()
        
        # cleanup the train/eval files
        for f in os.listdir(target_directory):
            if f.startswith("train_batch" or "eval_batch"): os.remove(target_directory + f)
        sleep(60)
    # end loop
    """
    precision_values = []
    recall_values = []
    accuracy_values = []
    # open csv files and collate results to produce mean and standard deviation for each error stat
    for f in os.listdir(csv_directory):
        if f.startswith("confusion_matrix"):
            with open(csv_directory + f, newline='') as csvfile:
                # setup reader function
                datareader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                try:
                    # skip headings
                    next(datareader)
                    # skip body of confusion matrix (we don't want to report these values - just the accuracy figures)
                    for skip in range(cifar10_eval.num_classes): next(datareader)
                    # read precision results
                    row = next(datareader)
                    precision_values.append([float(i) for i in row[1:]])
                    # read recall results
                    row = next(datareader)
                    recall_values.append([float(i) for i in row[1:]])
                    # read accuracy statistic
                    row = next(datareader)
                    accuracy_values.append(float(row[1]))            
                except StopIteration:
                    print("Attempted to read {} but file does not correspond with expected format".format(f))
                    break
            # unindent to close csv file
    # end loop

    # calculate mean and standard deviation for all accuracy stats
    accuracy_stats = numpy.mean(accuracy_values, axis=None), numpy.std(accuracy_values, axis=None)
    precision_stats = numpy.mean(precision_values, axis=0), numpy.std(precision_values, axis=0)
    recall_stats = numpy.mean(recall_values, axis=0), numpy.std(recall_values, axis=0)
    
    # print results and save to csv
    print(accuracy_stats)
    print(precision_stats)
    print(recall_stats)
    
    save_name = "Cross_Validation_{}_Sets".format(num_files)
    with open(csv_directory + save_name + ".csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # write headings
        datawriter.writerow(["-"] + ["Cat {}".format(i) for i in range(cifar10_eval.num_classes)])
        # add precision stats
        datawriter.writerow(["Precision Mean"] + precision_stats[0].tolist())
        datawriter.writerow(["Precision Std"] + precision_stats[1].tolist())
        datawriter.writerow(["Recall Mean"] + recall_stats[0].tolist())
        datawriter.writerow(["Recall Std"] + recall_stats[1].tolist())
        datawriter.writerow(["Accuracy Mean"] + [accuracy_stats[0]])
        datawriter.writerow(["Accuracy Std"] + [accuracy_stats[1]])
        
    
    print("Stats saved as {}.csv in {}".format(save_name, csv_directory))
    
### end Cross_Validation.get_stats