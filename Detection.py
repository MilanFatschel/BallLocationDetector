#Author:Milan Fatschel
#Date: 12/28/17

#This program tracks 1 ball and finds the location in the frame based on the color
#press 'q' to exit from the window

import cv2
import numpy as np
import imutils

#get the camera capture
cam = cv2.VideoCapture(0)

#hsv range values for a darker green (adjust for other colors)
colorLower = (60, 90, 6)
colorUpper = (95, 255, 255)
#type in which color the tracker is looking for
color = "Green"

while(True):
    #read in the capture frames as long as it exists
    frame_exists,frame = cam.read()
    
    #re-adjust frame size and convert frame to blur and hsv 
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #mask the frames for the color
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    #find the contours around the color
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]

    #initalize the center to none and get the number of contours
    center = None
    numContours = len(contours)

    #if there are multiple contours get the biggest, and draw a min circle around it to get the radius and center
    if numContours>0:
        contMax = max(contours,key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(contMax)
        center = (int(x),int(y))
        radius = int(radius)

    #draw a circle around the ball and draw a crosshair on it
        cv2.circle(frame,center,radius,(0,0,255),2)
        cv2.circle(frame, center, 2, (0, 0, 255),3)
        cv2.line(frame,(center[0]-radius,center[1]),(center[0]+radius,center[1]),(0,0,255),1)
        cv2.line(frame,(center[0],center[1]-radius),(center[0],center[1]+radius),(0,0,255),1)

    #if no contours set the center and radius to 0
    else:
        center = (0,0)
        radius = 0

    #set the text above the object
    cv2.putText(frame, "{} Ball X:{} Y:{}".format(color,center[0],center[1]),
                (center[0],center[1]-radius-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (0, 0, 255), 2)

    #show the resulting frames
    cv2.imshow("result",frame)

    #press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
cam.release()
cv2.destroyAllWindows()
