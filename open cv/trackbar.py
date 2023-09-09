# trackbar
# cv2.namedWindow('name of the window') creates a new window


import cv2
import numpy as np

def nothing(x) : # x is the current position of the trackbar
    print(x) #prints the value of the trackbar whenever its changed

img = np.zeros((300,512,3) , np.uint8)
cv2.namedWindow('newwindow')

cv2.createTrackbar('blue' , 'newwindow' , 0 , 255 , nothing) # secondlast and thirdlast arguments are initial and final values of the trackbar
cv2.createTrackbar('green' , 'newwindow' , 0 , 255 , nothing)
cv2.createTrackbar('red' , 'newwindow' , 0 , 255 , nothing)


while(1) :
    cv2.imshow('newwindow' , img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 :
        break

    b = cv2.getTrackbarPos('blue' , 'newwindow')
    g = cv2.getTrackbarPos('green', 'newwindow')
    r = cv2.getTrackbarPos('red', 'newwindow')

    img[:] = [b,g,r]

cv2.destroyAllWindows()

