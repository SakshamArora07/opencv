# add, split , size , dtype , split , merge
#cv2.add calculates the summ of two arrays/scalars etc (the size of the images should be same)

import cv2

img = cv2.imread('./photos/balls.jpg')
img2 = cv2.imread('./photos/dog.jpg')

print(img.shape) # returns a tuple of number of rows , columns and channels

print(img.size) # returns the total number of pixels accessed

print(img.dtype) # returns the image datatype obtained

b,g,r = cv2.split(img)

img = cv2.merge((b,g,r))

cv2.imshow('image' , img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ROI -region of interest and adding images

ball = img[50:100 , 70:120]
img[70:120 , 50:100] = ball # the coordinates are different but both sizes of the ball should match

cv2.imshow('balls image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.resize(img , (512,512))
img2 = cv2.resize(img2 , (512,512))

dest = cv2.add(img , img2)

cv2.imshow('added image' , dest)

cv2.waitKey(0)
cv2.destroyAllWindows()

#weighted add - we define which image has how much role (we decide which img is dominant)

weighted_dest = cv2.addWeighted(img , 0.2 , img2 , 0.8 , 0)
#0 is the gamma scalar value associated

cv2.imshow('weighted add' , weighted_dest)

cv2.waitKey(0)
cv2.destroyAllWindows()
