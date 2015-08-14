# get gray face with green lens overlayed - finished version 1
import numpy as np
import cv2
import sys
#import matplotlib.pyplot as plt
import cv2.cv as cv

print sys.argv[1],sys.argv[2]
PathForRoot = sys.argv[1] + '/assets'
VendorPath = '/app/vendor'
try:
    eyeI = cv2.imread(PathForRoot + '/green-big.png')
    releyeI = eyeI[9:82, 9:82, :]
    sizeye = np.shape(releyeI)
    eyerad = 36

    face_cascade = cv2.CascadeClassifier(VendorPath + '/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(VendorPath + '/share/OpenCV/haarcascades/haarcascade_eye.xml')

    #imgpath = '/Users/z001c3k/work/lens/testImages/'
    #imgoutpath = '/Users/z001c3k/work/lens/'
    #fname = 'image1.jpeg'
    fname = PathForRoot + "/CurrentImg.jpg"

    b_img = cv2.imread(fname)
    siz = np.shape(b_img)
    if np.float32(np.product(siz))/(640*640*3) < 1:
        scaleDownFactor = 1
        print "image resolution is lesser than desired value"
    else:
        scaleDownFactor = int(np.sqrt(np.float32(np.product(siz))/(640*640*3)))
    nr =int(siz[0]/scaleDownFactor)
    nc = int(siz[1]/scaleDownFactor)
    img = cv2.resize(b_img, (nc, nr))
    #gray_hres = cv2.cvtColor(b_img, cv2.COLOR_BGR2GRAY)
    gray_hres = b_img[:, :, 2]
    #orig_bw = (np.concatenate((gray_hres[:, :, np.newaxis], gray_hres[:, :, np.newaxis], gray_hres[:, :, np.newaxis]), axis = 2))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    allfaces = face_cascade.detectMultiScale(gray, 1.1, 5)
    s = np.shape(allfaces)
    if s[0]<1:
        print 'no faces detected'
    else:
        if s[0]==1:
            face = allfaces
        else:
            for i in np.arange(s[0]):
                faceFraction = float(allfaces[i,2]*allfaces[i,3])/(nr*nc)
                if (faceFraction>0.1) and (faceFraction<0.8):
                    face = allfaces[i,:] # this ensures there is only one face if at all detected
                    break        
        (x,y,w,h) = np.squeeze(face)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        (xhres,yhres,whres,hhres) = (int(scaleDownFactor*x),int(scaleDownFactor*y),int(scaleDownFactor*w),int(scaleDownFactor*h)) 
        roi_gray = gray[y:y+h, x:x+w]
        #roi_origbw = orig_bw[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        if np.shape(eyes)[0] == 0:
            print 'eye not detected'
        else:
            for (ex,ey,ew,eh) in eyes:
                eyecx = ey+eh/2
                if np.squeeze([eyecx>0.2*h] and [eyecx<0.5*h]):
                    #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    eye_det = roi_color[ey:ey+eh, ex:ex+ew]
                    #eye_det_origbw = roi_origbw[ey:ey+eh, ex:ex+ew]
                    (exhres,eyhres,ewhres,ehhres) = (int(scaleDownFactor*ex),int(scaleDownFactor*ey),int(scaleDownFactor*ew),int(scaleDownFactor*eh))
                    eye_det_gray_hres = gray_hres[yhres+eyhres:yhres+eyhres+ehhres, xhres+exhres:xhres+exhres+ewhres]
                    eye_det_b_img = b_img[yhres+eyhres:yhres+eyhres+ehhres, xhres+exhres:xhres+exhres+ewhres]
                    circles = cv2.HoughCircles(eye_det_gray_hres,cv.CV_HOUGH_GRADIENT,4,6,param1=70,param2=35,minRadius=int(ewhres/6.6),maxRadius=int(ewhres/5))
                    a = None
                    if (type(circles) == type(a)):
                        print ('one iris not detected')                            
                    else:
                        cenRow = int(circles[0,0,1])
                        cenCol = int(circles[0,0,0])
                        cenRad = int(circles[0,0,2])
                        dscalefac = np.float32(eyerad)/(cenRad)
                        nreye = int(sizeye[0]/dscalefac)
                        nceye = int(sizeye[1]/dscalefac)
                        smaleyeimg = cv2.resize(releyeI, (nceye, nreye))
                        ret1, mask1 = cv2.threshold(smaleyeimg[:,:,1], 5, 255, cv2.THRESH_BINARY)
                        nmask = cv2.bitwise_not(mask1)                    
                        iris_roi = eye_det_b_img[cenRow-cenRad:cenRow-cenRad+nreye, cenCol-cenRad:cenCol-cenRad+nceye, :] # removed a plus one
                        #iris_roi2 = iris_roi
                        iris_roi[:,:,0] = cv2.bitwise_and(iris_roi[:,:,0],iris_roi[:,:,0],mask = nmask)
                        iris_roi[:,:,1] = cv2.bitwise_and(iris_roi[:,:,1],iris_roi[:,:,1],mask = nmask)
                        iris_roi[:,:,2] = cv2.bitwise_and(iris_roi[:,:,2],iris_roi[:,:,2],mask = nmask)
                        iris_roi = cv2.add(iris_roi,smaleyeimg)
                        eye_det_b_img[cenRow-cenRad:cenRow-cenRad+nreye, cenCol-cenRad:cenCol-cenRad+nceye, :] = iris_roi# removed a plus one
                        
                        #cv2.circle(eye_det, (np.float32(circles[0,0,0]/scaleDownFactor), np.float32(circles[0,0,1]/scaleDownFactor)), np.float32(circles[0,0,2]/scaleDownFactor), (255,0,0), 2)                        
                    #cv2.circle(eye_det, (circles[0,0,0], circles[0,0,1]), 3, (0,0,255), 2)
                else:
                    print 'bad eye detected'
except Exception,e:
    fo = open(PathForRoot + '/' + sys.argv[3], "w+")
    fo.write(str(e))
    fo.close()
#orig_bw[]
cv2.imwrite(PathForRoot + '/' + sys.argv[2], b_img)
