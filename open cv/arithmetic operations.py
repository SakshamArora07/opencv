# arithmetic operations

import cv2

img1 = cv2.imread('./photos/dog.jpg')
img2 = cv2.imread('./photos/balls.jpg')

img1 = cv2.resize(img1 , (512,512))
img2 = cv2.resize(img2 , (512,512))


dest = cv2.add(img1, img2)

dest1 = img1 + img2
dest2 = img1 - img2

cv2.imshow('add1' , dest)
cv2.imshow('add2' , dest1)
cv2.imshow('sub1' , dest2)

cv2.waitKey(0)
cv2.destroyAllWindows()

#weighted add - we define which image has how much role (we decide which img is dominant)

weighted_dest = cv2.addWeighted(img1 , 0.2 , img2 , 0.8 , 0)
#0 is the gamma scalar value associated

cv2.imshow('weighted add' , weighted_dest)

cv2.waitKey(0)
cv2.destroyAllWindows()