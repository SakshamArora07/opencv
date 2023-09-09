# bitwise operations - useful while working with masks

import cv2
import numpy as np

img1 = cv2.imread('./photos/dog.jpg')
img2 = cv2.imread('./photos/balls.jpg')

img1 = cv2.resize(img1 , (512,512))
img2 = cv2.resize(img2 , (512,512))

bitAND = cv2.bitwise_and(img1 , img2)
cv2.imshow('and' , bitAND)

bitOR = cv2.bitwise_or(img1 , img2)
cv2.imshow('or' , bitOR)

bitXOR = cv2.bitwise_xor(img1 , img2)
cv2.imshow('xor' , bitXOR)

bitNOT = cv2.bitwise_not(img1)
cv2.imshow('not' , bitNOT)

cv2.waitKey(0)
cv2.destroyAllWindows()
