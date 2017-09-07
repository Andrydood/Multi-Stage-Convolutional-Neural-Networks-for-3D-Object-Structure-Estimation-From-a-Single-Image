# @Author: andreacasino
# @Date:   2017-07-20T18:29:13+01:00
# @Last modified by:   andreacasino
# @Last modified time: 2017-09-07T14:39:40+01:00

'''
Used to repurpose the data set into usable training and testing data
'''
import scipy.io as sio
import numpy as np
import cv2 as cv
import re
import rotateAndScale as rt
import sys

from settings import *

#Load correct keypoint coordinates
#These are in the form (D,K,P,N) where D is the 2D dimensions [column, row](width,height), K is the number of keypoints,
# P is the number of people tha tlabeled the image and N is what image it is

def calculateAverageCoords(loadedCoords):
    #The multiple labels of each image are then averaged unless they don't exist (tested by seeing if their first member is 0 or nan)
    coords = loadedCoords['coords']
    size = coords.shape
    averageCoords = np.zeros([size[0],size[1],size[3]])

    for img in range(size[3]):
        count = 0
        for set in range(size[2]):
            if coords[0,0,set,img] != 0 and ~(np.isnan(coords[0,0,set,img])):
                averageCoords[:,:,img] = averageCoords[:,:,img] + coords[:,:,set,img]
                count += 1

        averageCoords[:,:,img] = averageCoords[:,:,img]/count

    #The output will be in the form (D,K,N)
    return averageCoords


#Load images from given directory and find indices for each image corrisponding to the coords
def importImages(directory,inputCoords):
    fullDirectory = '../keypoint-5/'+directory
    images = []
    size = inputCoords.shape
    coords = np.zeros([size[0],size[1],size[2]])

    with open(fullDirectory) as f:
        for idx,line in enumerate(f):

            #Use regex to find series of 0s and then get the number after it
            fullCoord = line[-13:-5]
            fullNum = re.search('(0+)',fullCoord)
            numStart =  fullNum.span()[1]
            coordIndex = int(fullCoord[numStart:])
            coords[:,:,idx] = inputCoords[:,:,coordIndex-1]

            #Remove the \n and the .
            imageDirectory = '../keypoint-5'+line[1:-1]
            images.append(cv.imread(imageDirectory,1))

    #Truncate to remove excess space
    outputCoords = coords[:,:,0:len(images)]

    return (images,outputCoords)

#Resize images and put them in a numpy array
def resizeImages(images):

    resizedImages = np.zeros([HEIGHT,WIDTH,3,len(images)])

    for idx,image in enumerate(images):
        print(idx)
        resizedImages[:,:,:,idx] = imResizeWithBorders(image)

    return resizedImages


def imResizeWithBorders(image):
    imHeight = image.shape[0]
    imWidth = image.shape[1]

    if imWidth >= imHeight:
        localWidth = WIDTH
        localHeight = int(HEIGHT*(imHeight/imWidth))
        borderSize = (HEIGHT - localHeight)/2
        image = cv.resize(image,(localWidth,localHeight))
        image = cv.copyMakeBorder(image,int(borderSize),int(borderSize),0,0,cv.BORDER_REPLICATE)

    else:
        localHeight = HEIGHT
        localWidth = WIDTH * (imWidth/imHeight)
        borderSize = (WIDTH - localWidth)/2
        image = cv.resize(image,(int(localHeight),int(localWidth)))
        image = cv.copyMakeBorder(image,0,0,int(borderSize),int(borderSize),cv.BORDER_REPLICATE)

    #To account for any rounding errors
    image = cv.resize(image,(int(HEIGHT),int(WIDTH)))

    return image

#Find scaling factor between original image size and new image size and apply it to coordinates
def resizeCoords(originalImages,coords):

    for idx,image in enumerate(originalImages):

        oldHeight = image.shape[0]
        oldWidth = image.shape[1]

        if oldWidth >= oldHeight:
            newWidth = WIDTH
            newHeight = HEIGHT*(oldHeight/oldWidth)
            borderSize = (HEIGHT - newHeight)/2

            heightScale = newHeight/oldHeight
            widthScale = newWidth/oldWidth

            coords[1,:,idx] = heightScale*coords[1,:,idx]+borderSize
            coords[0,:,idx] = widthScale*coords[0,:,idx]

        else:
            newHeight = HEIGHT
            newWidth = WIDTH * (oldWidth/oldHeight)
            borderSize = (WIDTH - newWidth)/2

            heightScale = newHeight/oldHeight
            widthScale = newWidth/oldWidth

            coords[1,:,idx] = heightScale*coords[1,:,idx]
            coords[0,:,idx] = widthScale*coords[0,:,idx]+borderSize


    return coords

#Keypoints in the form [Image,Keypoint,Dimensions(0:w,1:h)]
#Make sure that when the coord is NaN, it is changed to 0
def fixCoords(coords):
    size = coords.shape

    for image in range(size[2]):
        for keypoint in range(size[1]):
            for dimension in range(size[0]):
                if(np.isnan(coords[dimension,keypoint,image])):
                    coords[dimension,keypoint,image] = 0

    return coords

#Old images in the form [Height,Width,Colors,Image]
#New images in the form [Image,Height,Width,Colors]
#Make it so the first axis is that of image#
def fixImages(inputImages):
    inputImages = np.swapaxes(inputImages,0,3);
    inputImages = np.swapaxes(inputImages,1,3);
    inputImages = np.swapaxes(inputImages,2,3);

    return inputImages

#Calculates probability of gaussian model with given parameters
def gaussianDistrib(x1,x2,mean1,mean2,sigma):
    return np.exp(-((x1 - mean1)*(x1 - mean1))/(2*sigma*sigma)-((x2 - mean2)*(x2 - mean2))/(2*sigma*sigma))

#Generate heatmaps out of the coordinates
def generateHeatmaps(coords):
    size = coords.shape
    imageRange = 40

    #In format(Image,keypoint,height,width)
    heatmaps = np.zeros((size[2],HEIGHT,WIDTH,KEYPOINTAMOUNT))
    for image in range(size[2]):

        for keypoint in range(KEYPOINTAMOUNT-1):
            meanWidth = int(coords[0,keypoint,image])
            meanHeight = int(coords[1,keypoint,image])

            #Only go in the vicinity of the keypoint
            for h in range(meanHeight-imageRange,meanHeight+imageRange+1):
                    if(h>=0 and h<HEIGHT):
                        for w in range(meanWidth-imageRange,meanWidth+imageRange+1):
                            if(w>=0 and w<WIDTH):
                                #Value equal to gaussian with mean at the given keypoint
                                heatmaps[image,h,w,keypoint] = gaussianDistrib(h,w,meanHeight,meanWidth,HEATMAPSIGMA)

        heatmaps[image,:,:,keypoint+1] = np.ones((HEIGHT,WIDTH)) - np.sum(heatmaps[image,:,:,:],2)

    return heatmaps

def flipImagesAndHeatmaps(images,heatmaps):

    size = heatmaps.shape

    images_new = np.zeros((size[0]*2,HEIGHT,WIDTH,3))
    heatmaps_new = np.zeros((size[0]*2,HEIGHT,WIDTH,KEYPOINTAMOUNT))

    for imageNum in range(size[0]):
        #Saving the original image and the flipped image
        images_new[imageNum*2,:,:,:] = images[imageNum,:,:,:]
        images_new[imageNum*2+1,:,:,:] = cv.flip( images[imageNum,:,:,:], 1 )

        for keypointNum in range(size[3]):
            #Doing the same for every keypoint
            currentHeatmap = heatmaps[imageNum,:,:,keypointNum]
            heatmaps_new[imageNum*2,:,:,keypointNum] = currentHeatmap
            heatmaps_new[imageNum*2+1,:,:,keypointNum] = cv.flip(currentHeatmap, 1 )

    return (images_new,heatmaps_new)



def angleImagesAndHeatmaps(images,heatmaps):

    size = heatmaps.shape

    images_new = np.zeros((size[0]*5,HEIGHT,WIDTH,3))
    heatmaps_new = np.zeros((size[0]*5,HEIGHT,WIDTH,KEYPOINTAMOUNT))

    for imageNum in range(size[0]):
        #Saving the original image and the rotated (and resized) images
        currentImage = images[imageNum,:,:,:]

        images_new[imageNum*5,:,:,:] = currentImage
        images_new[imageNum*5+1,:,:,:] = cv.resize(rt.rotateAndScaleExtend(currentImage, 15),(HEIGHT,WIDTH))
        images_new[imageNum*5+2,:,:,:] = cv.resize(rt.rotateAndScaleExtend(currentImage, -15),(HEIGHT,WIDTH))
        images_new[imageNum*5+3,:,:,:] = cv.resize(rt.rotateAndScaleExtend(currentImage, 30),(HEIGHT,WIDTH))
        images_new[imageNum*5+4,:,:,:] = cv.resize(rt.rotateAndScaleExtend(currentImage, -30),(HEIGHT,WIDTH))

        for keypointNum in range(size[3]-1):
            #Doing the same for every keypoint
            currentHeatmap = heatmaps[imageNum,:,:,keypointNum]

            heatmaps_new[imageNum*5,:,:,keypointNum] = currentHeatmap
            heatmaps_new[imageNum*5+1,:,:,keypointNum] = cv.resize(rt.rotateAndScaleBlack(currentHeatmap, 15),(HEIGHT,WIDTH))
            heatmaps_new[imageNum*5+2,:,:,keypointNum] = cv.resize(rt.rotateAndScaleBlack(currentHeatmap, -15),(HEIGHT,WIDTH))
            heatmaps_new[imageNum*5+3,:,:,keypointNum] = cv.resize(rt.rotateAndScaleBlack(currentHeatmap, 30),(HEIGHT,WIDTH))
            heatmaps_new[imageNum*5+4,:,:,keypointNum] = cv.resize(rt.rotateAndScaleBlack(currentHeatmap, -30),(HEIGHT,WIDTH))

        currentHeatmap = heatmaps[imageNum,:,:,keypointNum+1]
        heatmaps_new[imageNum*5,:,:,keypointNum+1] = currentHeatmap
        heatmaps_new[imageNum*5+1,:,:,keypointNum+1] = cv.resize(rt.rotateAndScaleWhite(currentHeatmap, 15),(HEIGHT,WIDTH))
        heatmaps_new[imageNum*5+2,:,:,keypointNum+1] = cv.resize(rt.rotateAndScaleWhite(currentHeatmap, -15),(HEIGHT,WIDTH))
        heatmaps_new[imageNum*5+3,:,:,keypointNum+1] = cv.resize(rt.rotateAndScaleWhite(currentHeatmap, 30),(HEIGHT,WIDTH))
        heatmaps_new[imageNum*5+4,:,:,keypointNum+1] = cv.resize(rt.rotateAndScaleWhite(currentHeatmap, -30),(HEIGHT,WIDTH))

    return (images_new,heatmaps_new)

#Saves image with keypoints shown
def showKeypoints(images,heatmaps):
    colors = [[255,0,0],[0,255,0],[0,0,255],[100,0,100],[0,100,100],[255,255,255],[255,255,0],[0,255,255],[255,0,255],[100,100,0],[0,100,0]]

    size = heatmaps.shape

    for idx in range(size[0]):

        currentHeatmap = np.zeros((HEIGHT,WIDTH,3))

        for kp in range(size[3]):

            currentHeatmap[:,:,0] = currentHeatmap[:,:,0]+np.multiply((heatmaps[idx,:,:,kp]/255),colors[kp][0])
            currentHeatmap[:,:,1] = currentHeatmap[:,:,1]+np.multiply((heatmaps[idx,:,:,kp]/255),colors[kp][1])
            currentHeatmap[:,:,2] = currentHeatmap[:,:,2]+np.multiply((heatmaps[idx,:,:,kp]/255),colors[kp][2])

        currentImage = np.multiply(images[idx,:,:,:],0.75)

        cv.imwrite("./data/outputImages/"+str(idx)+".jpg",currentImage + currentHeatmap )


def gatherImagesAndKeypoints(directory,funct):
    '''
    This function uses the piece of furniture specified in "directory" and gathers the images and coordinates
    found in that directory

    The funct variable is used to specify wether training ("train") or test("test") data is being gathered. It defaults to training

    Outputs ideal heatmap images in the form [Image,Height,Width,Keypoints]
    Outputs images in the form [Image,Height,Width,Colors]
    '''

    print("Generating "+funct+"ing data for "+directory+" images")

    print("Loading images...")
    #Import coordinates and find average amongst different point selections
    loadedCoords = sio.loadmat('../keypoint-5/'+directory+'/coords.mat')
    averageCoords = calculateAverageCoords(loadedCoords)

    #Load images and attach respective coordinate to each image
    (loadedImages,indexedCoords) = importImages(directory+'/'+funct+'.txt',averageCoords)
    print("Done")
    print("Resizing images...")
    #Resize images and coordinates to be of constant size
    images = resizeImages(loadedImages)
    coords = resizeCoords(loadedImages,indexedCoords)
    print("Done")

    #Turn the nans into 0s
    coords = fixCoords(indexedCoords)

    print("Generating heatmaps...")
    #Make ideal output heatmaps out of the coordinates
    heatmaps = generateHeatmaps(coords)
    print("Done")

    #Index the images correctly
    images = fixImages(images)


    if funct == 'train':

        #Make flipped and rotated images and add them to the regular images
        #Make 100 at a time and save 100 at a time (Done by applying the transforms to
        #10 images)

        print("Generating more training data and saving...")

        minRange = 0

        for imageRange in range(10,images.shape[0]+1,10):

            maxRange = imageRange

            print("Using images "+ str(minRange)+" to "+str(maxRange))

            currentImages = images[minRange:maxRange,:,:,:]
            currentHeatmaps = heatmaps[minRange:maxRange,:,:,:]

            (currentImages, currentHeatmaps) = flipImagesAndHeatmaps(currentImages,currentHeatmaps)
            (currentImages, currentHeatmaps) = angleImagesAndHeatmaps(currentImages,currentHeatmaps)


            np.save('./data/'+funct+'ing_'+directory+'/images_'+str(int((imageRange/10)-1)),currentImages)
            np.save('./data/'+funct+'ing_'+directory+'/heatmaps_'+str(int((imageRange/10)-1)),currentHeatmaps)

            minRange = maxRange

        print("Done")

    elif funct == 'test':

        print("Saving images and heatmaps...")


        np.save('./data/'+funct+'ing_'+directory+'/images',images)
        np.save('./data/'+funct+'ing_'+directory+'/heatmaps',heatmaps)

        print("Done")

def main():
    gatherImagesAndKeypoints(sys.argv[1],sys.argv[2])
    pass

if __name__ == '__main__':
    main()
