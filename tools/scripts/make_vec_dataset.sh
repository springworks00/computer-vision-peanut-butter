#!/bin/bash

# should be run in same directory as annotations file and vec file

/home/hborlik/projects/opencv/build/bin/opencv_createsamples -info ${1}.txt -vec ${1}.vec -num 500

# view the dataset
/home/hborlik/projects/opencv/build/bin/opencv_createsamples -vec ${1}.vec -w 500 -h 500