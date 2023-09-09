import cv2
import numpy as np

img = cv2.imread('photos/dog.jpg')
kernel = np.ones((51, 51)) / 51**2

minval, maxval, mincoord, maxcoord = cv2.minMaxLoc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

img = cv2.circle(img, maxcoord, 65, (255, 0, 0), 2)

cv2.imshow('bright spot image', img)
cv2.imwrite('Resources/retina_bright_spot.jpg', img)

cv2.waitKey(0)
cv2.destroyAllWindows()