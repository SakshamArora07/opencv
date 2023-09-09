# accessing image pixels and planes

import cv2

img = cv2.imread('./photos/dog.jpg')

pix = img[100,100] # gives the bgr values at the specified pixel
print(pix)

blue = img[ : , : ,0] # 0 : blue plane
green = img[ : , : , 1] # 1 : green plane
red = img[ : , : , 2] # 2 : red plane

print(blue)
print(green)
print(red)