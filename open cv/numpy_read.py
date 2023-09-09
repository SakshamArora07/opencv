# reading an image through numpy

import cv2
import numpy as np

# img = np.zeroes([size of the image , size of the image , color in number eg:3 for black] , datatype : np.uint8)
img = np.zeros([512,512,3] , np.uint8)

cv2.imshow('numpy img' , img)
cv2.waitKey(0)
cv2.destroyAllWindows()