import numpy as np
import cv2
import webbrowser as wb
import sys 

imgTest = cv2.imread('/app/assets/CurrentImg.jpg',0)
trainFnamesList = ['/app/python/diaper.png', '/app/python/chicco.png', '/app/python/justoneyou.png', '/app/python/minecraft.png', '/app/python/doritos.png'] 
urlList = ['http://m.target.com/s?searchTerm=diapers', 'http://m.target.com/s?searchTerm=chicco', 'http://m.target.com/s?searchTerm=just+one+you', 'http://m.target.com/s?searchTerm=minecraft', 'http://m.target.com/s?searchTerm=doritos']

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

s = [0, 0, 0, 0, 0]

src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches1[:50] ]).reshape(-1,1,2)
dst_pts = np.float32([ kpTest[m.trainIdx].pt for m in matches1[:50] ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
s[0] = np.abs(np.int16(M[0,1]))+np.abs(np.int16(M[1,0]))

src_pts = np.float32([ kp2[m.queryIdx].pt for m in matches2[:50] ]).reshape(-1,1,2)
dst_pts = np.float32([ kpTest[m.trainIdx].pt for m in matches2[:50] ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
s[1] = np.abs(np.int16(M[0,1]))+np.abs(np.int16(M[1,0]))

src_pts = np.float32([ kp3[m.queryIdx].pt for m in matches3[:50] ]).reshape(-1,1,2)
dst_pts = np.float32([ kpTest[m.trainIdx].pt for m in matches3[:50] ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
s[2] = np.abs(np.int16(M[0,1]))+np.abs(np.int16(M[1,0]))

src_pts = np.float32([ kp4[m.queryIdx].pt for m in matches4[:50] ]).reshape(-1,1,2)
dst_pts = np.float32([ kpTest[m.trainIdx].pt for m in matches4[:50] ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
s[3] = np.abs(np.int16(M[0,1]))+np.abs(np.int16(M[1,0]))

src_pts = np.float32([ kp5[m.queryIdx].pt for m in matches5[:50] ]).reshape(-1,1,2)
dst_pts = np.float32([ kpTest[m.trainIdx].pt for m in matches5[:50] ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
s[4] = np.abs(np.int16(M[0,1]))+np.abs(np.int16(M[1,0]))


#print urlList[np.argmin(s)]
fo = open('/app/assets/out.txt','w+')
fo.write(urlList[np.argmin(s)])
fo.close()