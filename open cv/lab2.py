#log transformation
import cv2
import numpy as np

img = cv2.imread('dog.jpg')
intermediate = 255/(np.log(1 + np.max(img))) #calculating the constant c
log_transform = intermediate * np.log(1+img) #applying log transform
log_transform = np.array(log_transform,dtype=np.uint8) #specifying the data type
cv2.imshow('log_transform' , log_transform)
cv2.imwrite('log_image.jpg', log_transform)
cv2.waitKey(0)
cv2.destroyAllWindows()

#negetation
import cv2

img = cv2.imread('dog.jpg')
b, g, r = cv2.split(img)
b = 255 - b
r = 255 - r
g = 255 - g
img_negate = cv2.merge((b, g, r))
cv2.imshow('Negated image', img_negate)
cv2.waitKey(100000)
cv2.destroyAllWindows()

#gamma correction
import cv2
import numpy as np

gamma_img = cv2.imread('dog.jpg')
for gamma in [0.1 , 0.3 , 0.4 , 0.5]:
    gamma_corrected = np.array(255*(gamma_img/255)**gamma , dtype = np.uint8)
cv2.imshow('ORIGINAL' , gamma_img)
cv2.imshow('gamma_correction' , gamma_corrected)

cv2.imwrite("GammaImage.jpg",gamma_corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()

#histogram equalisation

import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('dog.jpg',0)


equalized_img=cv2.equalizeHist(img)

hist_original=cv2.calcHist([img],[0],None,[255],[0, 255])

hist_equalized=cv2.calcHist([equalized_img],[0],None,[255],[0, 255])

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original Image')

axs[1, 0].plot(hist_original, color='blue')
axs[1, 0].set_title('original')
axs[1, 0].set_xlim([0, 256])

axs[0, 1].imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Equalized image')

axs[1, 1].plot(hist_equalized, color='green')
axs[1, 1].set_title('Histogram equalized')
axs[1, 1].set_xlim([0, 256])



plt.plot(hist_equalized, color='cyan')
plt.show()

#histogram matching

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_cumulative(histogram):
    cumulative = np.cumsum(histogram, dtype=np.float32)
    return cumulative / histogram.sum()


def get_hist_map(inp_histogram, ref_histogram):
    pix_map = np.zeros(256)

    k = 0
    for i, val in enumerate(inp_histogram):
        while k < 255 and ref_histogram[k] < val:
            k += 1

        pix_map[i] = k

    return np.array(pix_map, dtype=np.uint8)


def map_image(image, image_map):
    new_img = np.zeros_like(image, dtype=np.uint8)
    r, c = image.shape

    for i in range(r):
        for j in range(c):
            new_img[i, j] = image_map[image[i, j]]

    return new_img


def histogram_specification(inp_image, ref_image):
    inp_pixels = np.array(inp_image, dtype=np.uint8).ravel()
    ref_pixels = np.array(ref_image, dtype=np.uint8).ravel()

    inp_hist = np.histogram(inp_pixels, 255, [0, 255])[0]
    ref_hist = np.histogram(ref_pixels, 255, [0, 255])[0]

    inp_c = get_cumulative(inp_hist)
    ref_c = get_cumulative(ref_hist)

    img_map = get_hist_map(inp_c, ref_c)

    return map_image(inp_image, img_map)


inp_img = cv2.imread('./photos/dog.jpg', 0)
ref_img = cv2.imread('./photos/balls.jpg', 0)

new_img = histogram_specification(inp_img, ref_img)
cv2.imwrite('./photos/transformed_image.jpg', new_img)

fig, ax = plt.subplots(3, 2, figsize=(10, 10))

for i, (title, img) in enumerate([('Input', inp_img), ('Reference', ref_img), ('Histogram Matched', new_img)]):
    ax[i, 0].imshow(img, cmap='gray')
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])
    ax[i, 0].set_title(f'{title} Image')
    ax[i, 1].plot(cv2.calcHist([img], [0], None, [255], [0, 255]))
    ax[i, 1].set_title(f'Histogram - {title} Image')

plt.tight_layout()

plt.savefig('./photos/histogram_specification.jpg')
plt.show()

# clahe - adaptive histogram equalization

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./photos/dog.jpg', 0)

clahe = cv2.createCLAHE(clipLimit=5)
img_eq = clahe.apply(img)

hist_orig = cv2.calcHist([img], [0], None, [256], [0, 255])
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 255])

fig, ax = plt.subplots(2, 2, figsize=(10, 6))

ax[0, 0].imshow(img, cmap='gray')
ax[0, 0].set_title('Original Image')
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 1].plot(hist_orig, color='black')
ax[0, 1].set_title('Histogram(Original)')
ax[0, 1].set_xlim([0, 255])

ax[1, 0].imshow(img_eq, cmap='gray')
ax[1, 0].set_title('Adaptive Histogram Equalized Image')
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])
ax[1, 1].plot(hist_eq, color='black')
ax[1, 1].set_title('Histogram(Adaptive Equalized)')
ax[1, 1].set_xlim([0, 255])

plt.tight_layout()
plt.savefig('exercise2.jpg')
plt.show()




