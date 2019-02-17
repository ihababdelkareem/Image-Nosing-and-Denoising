import cv2
import os
import numpy as np
import random
import time

def getPaddedImage(image,windowSizeMax):
    final =np.zeros((image.shape[0]+int(windowSizeMax/2),image.shape[1]+int(windowSizeMax/2)),np.uint8)
    for i in range(int(windowSizeMax/2),image.shape[0]+int(windowSizeMax/2)):
        for j in range(int(windowSizeMax / 2), image.shape[1]+int(windowSizeMax/2)):
            final[i][j]=image[i-int(windowSizeMax / 2)][j-int(windowSizeMax / 2)]
    return final

def addSaltAndPepperNoise(image,prob):

    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def addGaussianNoise(image,mean,stdv):
    noise = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    noise = cv2.randn(noise,mean,stdv)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(int(image[i][j])+int(noise[i][j]) >=255):
                image[i][j]=255
            else:
                image[i][j]=int(image[i][j])+int(noise[i][j])
    return image

def readFromDataSet(index):
     if(os.path.exists(("TestData/"+str(index)+'.jpg'))):
         return cv2.imread(("TestData/"+str(index)+'.jpg'),0)
     else:
         return cv2.imread(("TestData/"+str(index)+'.png'),0)

def showAndWaitKey(name,image):
    cv2.imshow(str(name),image)
    cv2.waitKey()

def saveNoisyImage(image,innerNoiseDirectory,name):
    if(not os.path.exists('NoiseOutput/'+str(innerNoiseDirectory))):
        os.makedirs('NoiseOutput/'+str(innerNoiseDirectory))
    return cv2.imwrite(('NoiseOutput/'+str(innerNoiseDirectory)+'/'+str(name)+'.jpg'),image)

def saveEnhancedImage(image,innerDirectory,name):
    if(not os.path.exists('Enhanced/'+str(innerDirectory))):
        os.makedirs('Enhanced/'+str(innerDirectory))
    return cv2.imwrite(('Enhanced/'+str(innerDirectory)+'/'+str(name)+'.jpg'),image)

def getNoisyImage(path):
    return cv2.imread('NoiseOutput/'+str(path)+'.jpg',0)

def getMedianFilter(image,size):
    return cv2.medianBlur(image,size)

def getBoxFilter(image,boxSize):
    kernel = np.ones((boxSize, boxSize), np.float32) / (boxSize**2)
    return cv2.filter2D(image,-1,kernel)
def getGaussianFilter(image,size,sigma):
    return cv2.GaussianBlur(image,(size, size),sigma)

def getAdaptiveMedianFilter(image,windowSizeMax):
    padded = getPaddedImage(image,windowSizeMax)
    returnedImage = np.zeros((image.shape[0],image.shape[1]),np.uint8)
    for i in range(int(windowSizeMax/2),image.shape[0]+int(windowSizeMax/2)):
        for j in range(int(windowSizeMax / 2), image.shape[1]+int(windowSizeMax/2)):
            currentSize = 3
            while(currentSize<=windowSizeMax):
                box=padded[i-int(currentSize/2):i+1+int(currentSize/2),j-int(currentSize/2):j+1+int(currentSize/2)]
                median = np.median(box)
                min = np.min(box)
                max = np.max(box)
                A1=median - min
                A2=median - max
                if(A1>0 and A2< 0 ):
                    break
                else:
                    currentSize=currentSize+2
            if(currentSize==windowSizeMax+2):
                returnedImage[i-int(windowSizeMax/2)][j-int(windowSizeMax/2)]=padded[i][j]
            #level B
            else:
                B1 = int(padded[i][j]) - min
                B2 = int(padded[i][j]) - max
                if(B1>0 and B2<0):
                    returnedImage[i - int(windowSizeMax / 2)][j - int(windowSizeMax / 2)] = int(padded[i][j])
                else:
                    returnedImage[i - int(windowSizeMax / 2)][j - int(windowSizeMax / 2)] = median
    return returnedImage

def getBilateralFilter(image,box,sigmaColor,sigmaSpace):
    return cv2.bilateralFilter(image,box,sigmaColor,sigmaSpace)

def getLocalNoiseReductionFilter(image,noiseVariance,boxSize):
    padded = getPaddedImage(image, boxSize)
    returnedImage = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for i in range(int(boxSize / 2), image.shape[0] + int(boxSize / 2)):
        for j in range(int(boxSize / 2), image.shape[1] + int(boxSize / 2)):
            box = padded[i - int(boxSize / 2):i + 1 + int(boxSize / 2),j- int(boxSize/ 2):j + 1 + int(boxSize /2)]
            localMean = np.mean(box)
            localVar  = np.var(box)
            if(int(localVar)!=0):
                returnedImage[i - int(boxSize / 2)][j - int(boxSize / 2)] = int(padded[i][j]) - int((noiseVariance/localVar)*(padded[i][j] - localMean))
            else:
                returnedImage[i - int(boxSize / 2)][j - int(boxSize / 2)] = int(padded[i][j])
    return returnedImage


cam = cv2.VideoCapture(1)
mirror = True
while True:
    ret_val, img = cam.read()
    if mirror:
        img = cv2.flip(img, 1)
    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
spProbList = [0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.12,0.15]
meanList = [0,10,20,30,40]
stdDevList = [20,30]
filterDecisions = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[3,0],[3,1],[4,0],[4,1]]
noisyImageDirectories = []
for i in range(20):
    currentFilter  = filterDecisions[i]
    currentImage = readFromDataSet(i)
    cv2.imwrite('Gray/'+str(i)+'.jpg',currentImage)
    if(len(currentFilter)==1):
        spProb = spProbList[filterDecisions[i][0]]
        noisyImage = addSaltAndPepperNoise(currentImage,spProb)
        saveNoisyImage(noisyImage,'sp',str(spProb))
        noisyImageDirectories.append('sp/'+str(spProb))
    else:
         mean = meanList[filterDecisions[i][0]]
         stdDev = stdDevList[filterDecisions[i][1]]
         noisyImage = addGaussianNoise(currentImage,mean,stdDev)
         saveNameAs =  'm '+ str(mean)+', s ' + str(stdDev)
         saveNoisyImage(noisyImage,'gaussian',saveNameAs)
         noisyImageDirectories.append('gaussian/' + saveNameAs)
boxValues = [7,11,15] # for box , median , gaussian , local , bilateral windows
adaptiveMedianMaxWindow = 13
gaussianSigma = [10,25,40]
bilateralPairs = [[50,15],[30,20],[30,50]] # sigma color , sigma space
localNoiseReductionVar = [200]

preEnhancedImage = getNoisyImage(noisyImageDirectories[16])
showAndWaitKey('x',preEnhancedImage)
grayImage = cv2.imread('Gray/17.jpg',0)
showAndWaitKey('x',preEnhancedImage)
#averaging with 3 boxes first
for box in boxValues:
    print('box '+str(box))
    start = time.time()
    boxEnhancedImage = getBoxFilter(preEnhancedImage,box)
    saveNameAs = noisyImageDirectories[17].replace('/',' ') + ' box ' + str(box)
    saveEnhancedImage(boxEnhancedImage,'box-filter',saveNameAs)
    end = time.time()
    print(end - start)
    print('MSE : ', cv2.PSNR(grayImage, boxEnhancedImage))
    print(np.mean((grayImage, boxEnhancedImage))** 2)
    print('-------------------------------')
    print('median ' + str(box))
    start = time.time()
    medianEnhancedImage = getMedianFilter(preEnhancedImage,box)
    saveNameAs = noisyImageDirectories[17].replace('/',' ') + ' median ' + str(box)
    saveEnhancedImage(medianEnhancedImage,'median-filter',saveNameAs)
    end = time.time()
    print('MSE : ', cv2.PSNR(grayImage, medianEnhancedImage))
    print(np.mean((grayImage, medianEnhancedImage) )** 2)

    print(end - start)
    print('-------------------------------')
    for sigma in gaussianSigma:
        print('gaussian box ' + str(box) + ' sigma '+str(sigma))
        start = time.time()
        sigmaEnhancedImage = getGaussianFilter(preEnhancedImage,box,sigma)
        saveNameAs = noisyImageDirectories[17].replace('/', ' ') + ' box ' + str(box) + ' sigma ' + str(sigma)
        saveEnhancedImage(sigmaEnhancedImage, 'gaussian-filter', saveNameAs)
        end = time.time()
        print(end - start)
        print('MSE : ', cv2.PSNR(grayImage,sigmaEnhancedImage))
        print(np.mean((grayImage, sigmaEnhancedImage) )** 2)

        print('-------------------------------')

    for pair in bilateralPairs:
        print('bilateral box ' + str(box) + ' sigmas '+  str(pair[0]) + ' ' + str(pair[1]))
        start = time.time()
        bilateralEnhancedImage = getBilateralFilter(preEnhancedImage,box,pair[0],pair[1])
        saveNameAs = noisyImageDirectories[17].replace('/', ' ') + ' bilateral : box ' + str(box)+ ' sigmas '+  str(pair[0]) + ' ' + str(pair[1])
        saveEnhancedImage(bilateralEnhancedImage, 'bilateral-filter', saveNameAs)
        end = time.time()
        print('MSE : ', cv2.PSNR(grayImage, bilateralEnhancedImage))
        print(np.mean((grayImage, bilateralEnhancedImage) )** 2)
        print(end - start)
        print('-------------------------------')
    for var in localNoiseReductionVar:
        if(box == 11):
            print('local noise ' + str(box)+ ' noise var = 200')
            start = time.time()
            localNoiseReductionEnhancedImage = getLocalNoiseReductionFilter(preEnhancedImage,var,box)
            saveNameAs = noisyImageDirectories[17].replace('/', ' ') + ' localNoise : box ' + str(box)+ ' noise-var ' + str(var)
            saveEnhancedImage(localNoiseReductionEnhancedImage, 'local-noise-reduction-filter', saveNameAs)
            end = time.time()
            print(end - start)
            print('MSE : ', cv2.PSNR(grayImage, localNoiseReductionEnhancedImage))
            print(np.mean((grayImage, localNoiseReductionEnhancedImage) )** 2)
            print('-------------------------------')
print('adaptive median')
start = time.time()
adaptiveMedianEnhancedImage = getAdaptiveMedianFilter(preEnhancedImage,adaptiveMedianMaxWindow)
saveNameAs = noisyImageDirectories[17].replace('/', ' ') + ' adaptive-median maxBox ' + str(adaptiveMedianMaxWindow)
saveEnhancedImage(adaptiveMedianEnhancedImage, 'adaptive-median-filter', saveNameAs)
end = time.time()
print(end - start)
print('MSE : ', cv2.PSNR(grayImage, adaptiveMedianEnhancedImage))
print(np.mean((grayImage, adaptiveMedianEnhancedImage))**2)
print('-------------------------------')



