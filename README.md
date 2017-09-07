Dependencies: OpenCV, scipy.io, tensorflow, imutils

###Gather.py
Used to generate training data from keypoint-5 dataset

###Settings.py
Global variables

###cpm/train_model_init.py
Used to train the single stage keypoint identification network

###cpm/train_model_multi_stage.py
Used to train the multi stage keypoint identification network

###cpm/cpm.py
Contains keypoint identification network architecture

###cpm/multi_stage_evaluate.py
Allows to input one image and get the heatmap output

##3D_lifting/train_model.py
Used to train the 3D lifting network

###3D_lifting/network.py
Contains 3D lifting network architecture

###3D_lifting/evaluate_model.py
Allows lifting of 1 set of heatmaps to 3D

###3D_lifting/generateData.m
Used for manipulation and projection of 3D data

###3D_lifting/heatmapRegression.m
Used for isolating keypoints from n+1th image

###3D_lifting/testOutput.m
Tests network output

###3D_lifting/printChair.m
Draws skeleton of 3D keypoints
