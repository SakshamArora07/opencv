# gamma / power law transformation -  compress high intensity values and expand/brighten lower intensity values

import numpy as np
import cv2

img1 = cv2.imread('./photos/star.jpg' , 0)

gamma = 2
img2 = np.power(img1,gamma)

gamma = 3
img3 = np.power(img1,gamma)

gamma = 4
img4 = np.power(img1,gamma)

gamma = 8
img5 = np.power(img1,gamma)

cv2.imshow('original' , img1)
cv2.imshow('gamma1' , img2)
cv2.imshow('gamma2' , img3)
cv2.imshow('gamma3' , img4)
cv2.imshow('gamma4' , img5)

cv2.waitKey(0)
cv2.destroyAllWindows()  