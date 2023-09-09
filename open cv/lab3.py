# gaussian blur

import cv2
from matplotlib import pyplot as plt
import numpy as np
img1 = cv2.imread('./photos/noise.jpg' , 0)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img3 = cv2.GaussianBlur(img1, (5,5), 0)
cv2.imshow("original" , img1)
cv2.imshow("gaussian ", img3)
cv2.waitKey(0)

# box filter
img1 = cv2.imread('./photos/noise.jpg' , 0)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img3 = cv2.boxFilter(img1, -1, (5,5))
cv2.imshow("original" , img1)
cv2.imshow("box filter ", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()

#median filter

import cv2

# Load the image in grayscale
img = cv2.imread('./photos/noise.jpg', 0)

# Apply the median filter with a specified kernel size (e.g., 5x5)
median_filtered = cv2.medianBlur(img, 5)

# Display the original and filtered images
cv2.imshow("Original Image", img)
cv2.imshow("Median Filtered Image", median_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()


#unsharpening masking
import cv2
import numpy as np
def unsharp_mask(image, val):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharp = image - val * laplacian
    sharp[sharp > 255] = 255
    sharp[sharp < 0] = 0
    return sharp
cv2.waitKey(0)


img = cv2.imread('./photos/balls.jpg')
sharp_img = np.zeros_like(img)
for i in range(3):
    sharp_img[:, :, i] = unsharp_mask(img[:, :, i], 1.3)
cv2.imshow("This is the sharpened img",sharp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#sobel and laplacian for sharpening

img = cv2.imread('./photos/balls.jpg',cv2.IMREAD_GRAYSCALE)
assert img is not None,"FIle not read"

laplacian=cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
cv2.waitKey(0)


#gradient and phase angle of an image

image = cv2.imread('./photos/dog.jpg',0)
Gaussian=cv2.GaussianBlur(image,(7,7),0)
Sobely=cv2.Sobel(Gaussian,cv2.CV_64F,0,1,5)
Sobelx=cv2.Sobel(Gaussian,cv2.CV_64F,1,0,5)
Sobelxy=cv2.Sobel(Gaussian,cv2.CV_64F,1,1,5)
laplace=cv2.Laplacian(Gaussian,cv2.CV_64F)
sobel=np.hypot(Sobelx,Sobely)
phase=cv2.phase(Sobelx,Sobely)
(fig, axs) = plt.subplots(nrows=1, ncols=3, figsize=(3, 3))
axs[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
axs[0].set_title('Original IMAGE')
axs[1].imshow(sobel,cmap="gist_gray")
axs[1].set_title('Gradient')
axs[2].imshow(phase,cmap="gist_gray")
axs[2].set_title('Phase')
plt.show()
cv2.waitKey(0)





