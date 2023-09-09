#histograms

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./photos/dog.jpg')

cv2.imshow('image' , img)

#using matplotlib
plt.hist(img.ravel() , 256  , [0,256] ) #max number of pixel values and range of pixels)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

b,g,r = cv2.split(img)

cv2.imshow('blue' , b)
cv2.imshow('green' , g)
cv2.imshow('red' , r)

plt.hist(b.ravel() , 256  , [0,256] ) #max number of pixel values and range of pixels)
plt.hist(g.ravel() , 256  , [0,256] )
plt.hist(r.ravel() , 256  , [0,256] )
plt.show()


#using cv2

hist = cv2.calcHist([img] , [0] , None , [256] , [0 , 255])
#hist = cv2.calcHist([img - image] , [0] - channels [0,1,2] , None - if the image is in grayscale , [256] - no of bins , [0 , 255] - pixel range)
plt.plot(hist)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()