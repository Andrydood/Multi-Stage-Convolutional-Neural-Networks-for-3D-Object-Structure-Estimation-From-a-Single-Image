## Introduction

![](https://github.com/Andrydood/Multi-Stage-Convolutional-Neural-Networks-for-3D-Object-Structure-Estimation-From-a-Single-Image/blob/master/Screen%20Shot%202018-01-16%20at%2021.52.15.png?)

This was my final project for the Computer Graphics, Vision and Imaging MSc at UCL.

The goal of this project was to design a system can identify the 3D structure of an
object from a single RGB image. For the purpose of this implementation, images of chairs were used.
The system was designed to estimate the object structure by identifying a series of fundamental
keypoints within it, then lifting these into their respective 3D coordinates. The system consisted
of two key sections: a multi stage convolutional neural network for keypoint extraction and a fully
connected neural network to find the depth for the estimated 2D keypoint coordinates. The first
section was designed as an extension of the keypoint estimator found in Tome et al.[1] as well as
in Wei et al.[2], which is used for human keypoint identification. The fully connected network was
an implementation of the one found in Srivastava et al.[3].

During the evaluation of the system, the trained keypoint extractor was able to estimate
the coordinates of the keypoints on the object. Unfortunately, the system was not able to differentiate
between different keypoints on the same level of the chair (upper back, seat and legs). However,
the project proved that there can be great improvements to an output if a multi stage achitecture
is used in convolutional neural networks instead of a single stage architecture. In addition, the
depth estimator was successfully implemented with positive results. Finally, the combined system was able to output a 3D model from an image input under the circumstance where the image portrays the chair facing the
camera. In this case, the system is able to exploit the known relationships between
the object and its frame to successfully estimate which keypoint belonged to which appendage.
The results of this were successfull as can be seen in the above image.

The full write up can be found in the pdf on this repository.

![](https://github.com/Andrydood/Multi-Stage-Convolutional-Neural-Networks-for-3D-Object-Structure-Estimation-From-a-Single-Image/blob/master/Screen%20Shot%202018-01-16%20at%2021.52.28.png?)

[1] D. Tomè, C. Russell, and L. Agapito, “Lifting from the deep: Convolutional 3d pose estimation
from a single image,” CoRR, vol. abs/1701.00295, 2017.

[2] S. Wei, V. Ramakrishna, T. Kanade, and Y. Sheikh, “Convolutional pose machines,” CoRR,
vol. abs/1602.00134, 2016.

[3] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A simple
way to prevent neural networks from overfitting,” Journal of Machine Learning Research,
vol. 15, pp. 1929–1958, 2014.

## Included Files

Dependencies: OpenCV, scipy.io, tensorflow, imutils

#### Gather.py
Used to generate training data from keypoint-5 dataset

#### Settings.py
Global variables

#### cpm/train_model_init.py
Used to train the single stage keypoint identification network

### cpm/train_model_multi_stage.py
Used to train the multi stage keypoint identification network

#### cpm/cpm.py
Contains keypoint identification network architecture

#### cpm/multi_stage_evaluate.py
Allows to input one image and get the heatmap output

#### 3D_lifting/train_model.py
Used to train the 3D lifting network

#### 3D_lifting/network.py
Contains 3D lifting network architecture

#### 3D_lifting/evaluate_model.py
Allows lifting of 1 set of heatmaps to 3D

#### 3D_lifting/generateData.m
Used for manipulation and projection of 3D data

#### 3D_lifting/heatmapRegression.m
Used for isolating keypoints from n+1th image

#### 3D_lifting/testOutput.m
Tests network output

#### 3D_lifting/printChair.m
Draws skeleton of 3D keypoints
