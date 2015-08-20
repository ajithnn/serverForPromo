import numpy as np
import cv2
import webbrowser as wb
import sys 

imgTest = cv2.imread('/app/assets/CurrentImg.jpg',0)
qImgSz = np.shape(imgTest)
scaleFac = np.sqrt(np.float(qImgSz[0])*qImgSz[1]/921600)
imgTest = cv2.resize(imgTest, (0,0), fx=1/scaleFac, fy=1/scaleFac)
trainFnamesList = ['/app/python/diaper.png', '/app/python/chicco.png', '/app/python/justoneyou.png', '/app/python/minecraft.png', '/app/python/doritos.png'] 
urlList = ['http://m.target.com/s?searchTerm=diapers', 'http://m.target.com/s?searchTerm=chicco', 'http://m.target.com/s?searchTerm=just+one+you', 'http://m.target.com/s?searchTerm=minecraft', 'http://m.target.com/s?searchTerm=doritos']

trainImlist = [cv2.imread(trainFnamesList[0],0)]
numTrainImgs = len(trainFnamesList)
for i in range(1,numTrainImgs):
    trainImlist.append(cv2.imread(trainFnamesList[i],0))

# Initiate SIFT detector
orb = cv2.ORB()
# find the keypoints and descriptors with SIFT
kpTrain = []
desTrain = []
kpTemp, desTemp = orb.detectAndCompute(trainImlist[0],None)
kpTrain = [kpTemp]
desTrain = [desTemp]

for i in range(1,numTrainImgs):
    kpTemp, desTemp = orb.detectAndCompute(trainImlist[i],None)
    kpTrain.append(kpTemp)
    desTrain.append(desTemp)

#RimgTest = cv2.resize(imgTest, (900, 1100))
kpTest, desTest = orb.detectAndCompute(imgTest,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = []
numMtchs = 50
matchesTemp = bf.match(desTrain[0],desTest)
matchesTemp = sorted(matchesTemp, key = lambda x:x.distance)
matches = [matchesTemp]
for i in range(1,numTrainImgs):
    matchesTemp = bf.match(desTrain[i],desTest)
    matchesTemp = sorted(matchesTemp, key = lambda x:x.distance)
    matches.append(matchesTemp)

# Sort them in the order of their distance.
s = []
src_pts = []
dst_pts = []
M = []
mask = []
src_ptsTemp = np.float32([ kpTrain[0][m.queryIdx].pt for m in matches[0][:numMtchs] ]).reshape(-1,1,2)
dst_ptsTemp = np.float32([ kpTest[m.trainIdx].pt for m in matches[0][:numMtchs] ]).reshape(-1,1,2)
MTemp, maskTemp = cv2.findHomography(src_ptsTemp, dst_ptsTemp, cv2.RANSAC,5.0)
#sTemp = np.abs((MTemp[0,1]))+np.abs((MTemp[1,0]))


src_pts = [src_ptsTemp]
dst_pts = [dst_ptsTemp]
M = [MTemp]
mask = [maskTemp]
sTemp = np.sum(mask[0])
s = [sTemp]

for i in range(1,numTrainImgs):
    src_ptsTemp = np.float32([ kpTrain[i][m.queryIdx].pt for m in matches[i][:numMtchs] ]).reshape(-1,1,2)
    dst_ptsTemp = np.float32([ kpTest[m.trainIdx].pt for m in matches[i][:numMtchs] ]).reshape(-1,1,2)
    MTemp, maskTemp = cv2.findHomography(src_ptsTemp, dst_ptsTemp, cv2.RANSAC,5.0)    
    src_pts.append(src_ptsTemp)
    dst_pts.append(dst_ptsTemp)
    M.append(MTemp)
    mask.append(maskTemp)
    sTemp = np.sum(mask[i])
    s.append(sTemp)
    
#print urlList[np.argmin(s)]
if ((np.max(s)/numMtchs) > 0.5):
    fo = open('/app/assets/out.txt','w+')
    fo.write(urlList[np.argmin(s)])
    fo.close()
else:
    fo = open('/app/assets/out.txt','w+')
    fo.write('Not Found')
    fo.close()	
