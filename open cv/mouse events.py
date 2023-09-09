# mouse events - coordinates marking and bgr composition marking of all points

import cv2
import numpy as np

#iterating over all the functions/events of cv2 package
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

def click_event(event , x , y , flags , param) : #x and y are coordinates
    if event == cv2.EVENT_LBUTTONDOWN : # left click gives the coordinates of that point
        print(x , ' , ' , y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ' , ' + str(y)
        cv2.putText(img , strXY , (x,y) , font , 0.5 , (255,255,0) , 2)
        cv2.imshow('image' , img)

    if event == cv2.EVENT_RBUTTONDOWN : #right click gives the bgr composition at that point
        blue = img[y,x,0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue) + ' , ' + str(green)+ ' , ' + str(red)
        cv2.putText(img , strBGR , (x,y) , font , 0.5 , (255,255,0) , 2)
        cv2.imshow('image' , img)

# img = np.zeros([512,512,3] , np.uint8)
img = cv2.imread('./photos/dog.jpg' )
cv2.imshow('image' , img)

cv2.setMouseCallback('image' , click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()