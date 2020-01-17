# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:44:01 2019

@author: Nanda Kishore Mallapragada
"""

import numpy as np
import cv2
import math

webCam = cv2.VideoCapture(0)
#bgnd = cv2.createBackgroundSubtractorKNN()
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
while(1):
    try:
        ret,frame = webCam.read()
        frame = cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        handInput=frame[100:300, 100:300]
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
        grayImage = cv2.cvtColor(handInput, cv2.COLOR_BGR2GRAY)
        
#        lower_skin = np.array([0,200], dtype=np.uint8)
#        upper_skin = np.array([55,0.8*255], dtype=np.uint8)
        
#        edged = cv2.Canny(grayImage,30,200)
#        
#        lower_skin = np.array([0,0.05*255,0.1*255], dtype=np.uint8)
#        upper_skin = np.array([15,0.6*255,0.8*255], dtype=np.uint8)
#         
#        mask = cv2.inRange(grayImage, lower_skin, upper_skin)
        mask = cv2.erode(grayImage,kernel,iterations = 3)
#        mask = cv2.dilate(mask,kernel,iterations = 4)
        
        blurHand = cv2.GaussianBlur(mask,(5,5),0) 
        

        ret,thresh1 = cv2.threshold(blurHand,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        _, contours, hrcy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#        print(hrcy)
        
#        contour1 = max(contours,key = lambda x: cv2.contourArea(x))
#        print(contour1, len(contours))
        max_area =0
        for i in range(len(contours)):
            contour1 = contours[i]
            area = cv2.contourArea(contour1)
            if(area>max_area):
                max_area = area
                ci =i
        cnt = contours[ci]
#        epsilon = 0.0005*cv2.arcLength(cnt,True)
#        approx= cv2.approxPolyDP(cnt,epsilon,True)
        
        drawHull = cv2.convexHull(cnt)
#        print(drawHull.shape)
        areahull = cv2.contourArea(drawHull)
        areacnt = cv2.contourArea(cnt)
        arearatio=((areahull-areacnt)/areacnt)*100
        
#        moments = cv2.moments(cnt)
#        if moments['m00']!=0:
#                cx = int(moments['m10']/moments['m00']) # cx = M10/M00
#                cy = int(moments['m01']/moments['m00']) # cy = M01/M00
#              
#        centr=(cx,cy)       
#        cv2.circle(img,centr,5,[0,0,255],2)           
#        print(drawHull.shape)
        contDraw = np.zeros(frame.shape,np.uint8)
        cv2.drawContours(contDraw,[cnt],-1,(0,255,0),2)
        cv2.drawContours(contDraw,[drawHull],-1,(0,0,255),2)
        
        cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
       
        drawDefects = cv2.convexHull(cnt,returnPoints = False)
        
        defects = cv2.convexityDefects(cnt,drawDefects)
#        print(defects.shape)
        mind=0
        maxd=0
        i=0
        l=0
#        for i in range(defects.shape[0]):
#            s,e,f,d = defects[i,0]
#            start = tuple(cnt[s][0])
#            end = tuple(cnt[e][0])
#            far = tuple(cnt[f][0])
##            centr = (70,90)
#            dist = cv2.pointPolygonTest(cnt,centr,True)
#            cv2.line(contDraw,start,end,[0,255,0],2)                
#            cv2.circle(contDraw,far,5,[255,0,0],-1)
#        print(i)
        
#        l=0
#        
#    #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            pt= (100,180)
            
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
#            dist = cv2.pointPolygonTest(cnt,centr,True)
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(contDraw, far, 3, [255,0,0], -1)
            
            #draw lines around hand
            cv2.line(contDraw,start, end, [0,255,0], 2)
#        dist = cv2.pointPolygonTest(cnt,centr,True)
#            cv2.line(contDraw,start,end,[0,255,0],2) 
        l += 1
        print(l)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            if areacnt<4000:
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                if arearatio<16:
                    cv2.putText(frame,'Fist',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)                   
                else:
                    cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif l==2:
            cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==3:
         
            cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif l==4:
            cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==5:
            cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        cv2.imshow('draw1',contDraw)
#        print(drawHull)
        cv2.imshow('mask',thresh1)
        cv2.imshow('frame',frame)
    except:
        pass
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
webCam.release()



