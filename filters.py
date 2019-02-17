def getPaddedImage(image,windowSizeMax):
    final =np.zeros((image.shape[0]+int(windowSizeMax/2),image.shape[1]+int(windowSizeMax/2)),np.uint8)
    for i in range(int(windowSizeMax/2),image.shape[0]+int(windowSizeMax/2)):
        for j in range(int(windowSizeMax / 2), image.shape[1]+int(windowSizeMax/2)):
            final[i][j]=image[i-int(windowSizeMax / 2)][j-int(windowSizeMax / 2)]
    return final
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
