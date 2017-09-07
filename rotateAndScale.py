# @Author: andreacasino
# @Date:   2017-07-20T18:29:13+01:00
# @Last modified by:   andreacasino
# @Last modified time: 2017-08-25T17:59:59+01:00



#from: https://stackoverflow.com/questions/11764575/python-2-7-3-opencv-2-4-after-rotation-window-doesnt-fit-image

import cv2 as cv2
import numpy as np
from settings import *

def rotateAndScaleBlack(img, degreesCCW):
    newHeight = HEIGHT*2
    newWidth = WIDTH*2

    #Pad Image
    paddedImage = cv2.copyMakeBorder(img,int(HEIGHT/2),int(HEIGHT/2),int(WIDTH/2),int(WIDTH/2),cv2.BORDER_CONSTANT,value=0)

    #Rotate Image
    M = cv2.getRotationMatrix2D((newWidth/2,newHeight/2),degreesCCW,1)
    rotatedImage = cv2.warpAffine(paddedImage,M,(newWidth,newHeight))

    #Crop image back to original size
    croppedImage = rotatedImage[int(HEIGHT/3):int(-(HEIGHT/3)),int(WIDTH/3):int(-(WIDTH/3))]

    outputImage = cv2.resize(croppedImage,(HEIGHT,WIDTH))

    return outputImage

def rotateAndScaleWhite(img, degreesCCW):
    newHeight = HEIGHT*2
    newWidth = WIDTH*2

    #Pad Image
    paddedImage = cv2.copyMakeBorder(img,int(HEIGHT/2),int(HEIGHT/2),int(WIDTH/2),int(WIDTH/2),cv2.BORDER_CONSTANT,value=1)

    #Rotate Image
    M = cv2.getRotationMatrix2D((newWidth/2,newHeight/2),degreesCCW,1)
    rotatedImage = cv2.warpAffine(paddedImage,M,(newWidth,newHeight))

    #Crop image back to original size
    croppedImage = rotatedImage[int(HEIGHT/3):int(-(HEIGHT/3)),int(WIDTH/3):int(-(WIDTH/3))]

    outputImage = cv2.resize(croppedImage,(HEIGHT,WIDTH))

    return outputImage

def rotateAndScaleExtend(img, degreesCCW):

    newHeight = HEIGHT*2
    newWidth = WIDTH*2

    #Pad Image
    paddedImage = cv2.copyMakeBorder(img,int(HEIGHT/2),int(HEIGHT/2),int(WIDTH/2),int(WIDTH/2),cv2.BORDER_REPLICATE)

    #Rotate Image
    M = cv2.getRotationMatrix2D((newWidth/2,newHeight/2),degreesCCW,1)
    rotatedImage = cv2.warpAffine(paddedImage,M,(newWidth,newHeight))

    #Crop image back to original size
    croppedImage = rotatedImage[int(HEIGHT/3):int(-(HEIGHT/3)),int(WIDTH/3):int(-(WIDTH/3)),:]

    outputImage = cv2.resize(croppedImage,(HEIGHT,WIDTH))

    return outputImage
