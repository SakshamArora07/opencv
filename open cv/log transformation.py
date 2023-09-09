# log transformation  - brightens the low intensity values
#                     - helps in viewing details in a black/dull photo
#                     - compress high intensity values and expand/brighten lower intensity values

import numpy as np
import cv2

img1 = cv2.imread('./photos/star.jpg' , 0)

img2 = np.uint8(np.log1p(img1)) # log1p finds log(1+r)

thresh = 1
img3 = cv2.threshold(img2,thresh,255,cv2.THRESH_BINARY)[1] #converting the image into binary image to view final log transform output image.

cv2.imshow('original' , img1)

cv2.imshow('log transform 1' , img3)

cv2.waitKey(0)
cv2.destroyAllWindows()