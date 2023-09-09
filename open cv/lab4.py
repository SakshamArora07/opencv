#HOUGH TRANSFORM

import numpy as np
import cv2
import matplotlib.pyplot as plt

import cv2

plt.figure(figsize=(8, 6))

img = cv2.imread('./photos/chess.png')

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.title('Original Image')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200, apertureSize=3) # for horizontal and vertical edges
# edges = cv2.Canny(gray, 50, 150, apertureSize=3) # for diagonal edges

# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 127, minLineLength=100, maxLineGap=10) # for horizontal and vertical edge
lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 100, minLineLength=100, maxLineGap=10) # for slant edges


if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 15)

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.title('Hough Transform')

plt.tight_layout()
plt.savefig('line_hough.png')
plt.show()


#THRESHOLDING

import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./photos/thumb.jpg', 0)

thresholds = [('Binary Thresholding', cv2.THRESH_BINARY),
              ('Binary Inverted Thresholding', cv2.THRESH_BINARY_INV),
              ('Truncated Thresholding', cv2.THRESH_TRUNC),
              ('Set to 0 Thresholding', cv2.THRESH_TOZERO),
              ('Set to 0 inverted Thresholding', cv2.THRESH_TOZERO_INV)]

plt.figure(figsize=(8, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Original Image')

for i, (name, threshold) in enumerate(thresholds):
    ret, threshold_image = cv2.threshold(image, 127, 255, threshold) # threshold value = 127
    plt.subplot(2, 3, i + 2)
    plt.imshow(threshold_image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(name)

plt.tight_layout()
plt.savefig('thresholdingbuiltin.png')
plt.show()


#canny edge detection

image = cv2.imread('./photos/dog.jpg', 0)
edges = cv2.Canny(image, 100, 200)
cv2.imshow("Original Image",image)
cv2.imshow("Canny ", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#CANNY EDGE DETECTION FROM SCRATCH

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Canny_edge_detector(img,thlow = None,thhigh = None):
    # 1- Gaussian Smoothing
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = cv2.GaussianBlur(gimg,(5,5),1.4)

    #2 - Compute Gradient
    gradient_x = cv2.Sobel(np.float32(gimg),cv2.CV_64F, 1, 0, 3)
    gradient_y = cv2.Sobel(np.float32(gimg),cv2.CV_64F, 0, 1, 3)

    #3- Compute Magnitude and Direction
    mag,dir = cv2.cartToPolar(gradient_x,gradient_y,angleInDegrees=True)

    # Setting minimum and maximum thresholds for Hysterisis Thresholding
    mag_max = np.max(mag)
    if not thlow: thlow = mag_max * 0.1
    if not thhigh: thhigh = mag_max  * 0.5

    height, width = gimg.shape

    #For every pixel:
    for i_x in range(width):
        for i_y in range(height):
            grad_ang = dir[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # X-axis
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

                # top right (diagonal-1)
            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

                # Y-axis
            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y - 1
                neighb_2_x, neighb_2_y = i_x, i_y + 1

                # top left (diagonal-2) direction
            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y - 1

                # Now it restarts the cycle
            elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

                # Non-maximum suppression step
            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x] = 0
                    continue

            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x] = 0

    weak_ids = np.zeros_like(gimg)
    strong_ids = np.zeros_like(gimg)
    ids = np.zeros_like(gimg)

    # Hysterisis thresholding step
    for i_x in range(width):
        for i_y in range(height):

            grad_mag = mag[i_y, i_x]

            if grad_mag < thlow:
                mag[i_y, i_x] = 0
            elif thhigh > grad_mag >= thlow:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2

    return mag


frame = cv2.imread('./photos/dog.jpg')

canny_img = Canny_edge_detector(frame)

# Displaying the input and output image
cv2.imshow("Canny",canny_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#thresholding scratch

import cv2
import numpy as np
import matplotlib.pyplot as plt


def binary_thresholding(image, val):
    image = np.array(image)
    out_img = np.zeros(image.shape)
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            if image[i, j] >= val:
                out_img[i, j] = 255
    return out_img


def inv_binary_thresholding(image, val):
    out_img = binary_thresholding(image, val)
    out_img = 255 - out_img
    return out_img


def truncated_thresholding(image, val):
    out_img = np.copy(image)
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            if image[i, j] >= val:
                out_img[i, j] = val
    return out_img


def thresh_to_zero(image, val):
    out_img = np.copy(image)
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            if image[i, j] < val:
                out_img[i, j] = 0
    return out_img

def thresh_to_zero_inv(image, val):
    out_img = np.copy(image)
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            if image[i, j] >= val:
                out_img[i, j] = 0
    return out_img


image = cv2.imread('./photos/dog.jpg', 0)

thresholds = [('Binary Thresholding', binary_thresholding),
              ('Binary Inverted Thresholding', inv_binary_thresholding),
              ('Truncated Thresholding', truncated_thresholding),
              ('Set to 0 Thresholding', thresh_to_zero),
              ('Set to 0 inverted Thresholding', thresh_to_zero_inv)]

plt.figure(figsize=(10, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Original Image')

for i, (name, threshold) in enumerate(thresholds):
    threshold_image = threshold(image, 127)
    plt.subplot(2, 3, i + 2)
    plt.imshow(threshold_image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(name)

plt.tight_layout()
plt.savefig('photos/thresholdingscratch.png')
plt.show()