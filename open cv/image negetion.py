# image negetion

import cv2

img1 = cv2.imread('./photos/xray.jpg')
img2 = 255 - img1

cv2.imshow('xray' , img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

 
cv2.imshow('xray negetion' , img2)

cv2.waitKey(0)
cv2.destroyAllWindows()