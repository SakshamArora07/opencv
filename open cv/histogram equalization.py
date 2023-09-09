# histogram equalization - contrast and image enhancement
#                         - map intensities to new intensities based on CDF

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./photos/dog.jpg' , 0)
equalizedimg = cv2.equalizeHist(img)

hist1 = cv2.calcHist([img] , [0] , None , [255] , [0,255])
hist2 = cv2.calcHist([equalizedimg] , [0] , None , [255] , [0,255])

plt.plot(hist1  , color='red')
plt.plot(hist2 , color='cyan')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

#plotting everything together using subplots

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original Image')

axs[1, 0].plot(hist1, color='blue')
axs[1, 0].set_title('original')
axs[1, 0].set_xlim([0, 256])

axs[0, 1].imshow(cv2.cvtColor(equalizedimg, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Equalized image')

axs[1, 1].plot(hist2, color='green')
axs[1, 1].set_title('Histogram equalized')
axs[1, 1].set_xlim([0, 256])

plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()

# CLAHE - contrast limited adaptive histogram equalization

clahe = cv2.createCLAHE(clipLimit=2.0 , tileGridSize = (8,8))
cl1 = clahe.apply(img)
cv2.imshow('clahe img' , cl1)

hist = cv2.calcHist([cl1], [0], None, [256], [0, 255])
# hist = cv2.calcHist([img - image] , [0] - channels [0,1,2] , None - if the image is in grayscale , [256] - no of bins , [0 , 255] - pixel range)
plt.plot(hist)
plt.show()



