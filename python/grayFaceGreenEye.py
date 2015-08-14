import numpy as np
import cv2
#from matplotlib import pyplot as plt

imgTest = cv2.imread('/app/python/IMG5.jpg',0)
trainFnamesList = ['/app/python/diaper.png', '/app/python/chicco.png', '/app/python/justoneyou.png', '/app/python/minecraft.png', '/app/python/doritos.png'] 
urlList = ['http://www.target.com/s?searchTerm=diapers', 'http://www.target.com/s?searchTerm=chicco', 'http://www.target.com/s?searchTerm=just+one+you', 'http://www.target.com/s?searchTerm=minecraft', 'http://www.target.com/s?searchTerm=doritos']

img1 = cv2.imread(trainFnamesList[0],0)
img2 = cv2.imread(trainFnamesList[1],0)
img3 = cv2.imread(trainFnamesList[2],0)
img4 = cv2.imread(trainFnamesList[3],0)
img5 = cv2.imread(trainFnamesList[4],0)

trainImlist = [img1, img2, img3, img4, img5]

# Initiate SIFT detector
orb = cv2.ORB()

# find the keypoints and descriptors with SIFT

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
kp3, des3 = orb.detectAndCompute(img3,None)
kp4, des4 = orb.detectAndCompute(img4,None)
kp5, des5 = orb.detectAndCompute(img5,None)

kpTest, desTest = orb.detectAndCompute(imgTest,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches1 = bf.match(des1,desTest)
matches2 = bf.match(des2,desTest)
matches3 = bf.match(des3,desTest)
matches4 = bf.match(des4,desTest)
matches5 = bf.match(des5,desTest)

matches1 = sorted(matches1, key = lambda x:x.distance)
matches2 = sorted(matches2, key = lambda x:x.distance)
matches3 = sorted(matches3, key = lambda x:x.distance)
matches4 = sorted(matches4, key = lambda x:x.distance)
matches5 = sorted(matches5, key = lambda x:x.distance)

# Sort them in the order of their distance.
s = [0, 0, 0, 0, 0]

print matches1

for i in range(0,150):
    s[0] = s[0]+matches1[i].distance
    
for i in range(0,150):
    s[1] = s[1]+matches2[i].distance

for i in range(0,150):
    s[2] = s[2]+matches3[i].distance

for i in range(0,150):
    s[3] = s[3]+matches4[i].distance

for i in range(0,150):
    s[4] = s[4]+matches5[i].distance

print urlList[np.argmin(s)]