import cv2
import numpy as np

#reading a colored image
img_colored = cv2.imread('dog.jpg')
cv2.imshow('colored image' , img_colored )
cv2.waitKey(0)#0 captures a photo and 1 is for video
cv2.destroyAllWindows()



#reading a colored image as a grayscale image
img_grayscale = cv2.imread('dog.jpg' , 0)
cv2.imshow('grayscale image' , img_grayscale)
cv2.waitKey(0)
cv2.destroyAllWindows()

#reading a colored image and converting it to black and white permanently through write
img_convert = cv2.imread('dog.jpg' , 0)
cv2.imwrite('C:/Users/LAB/PycharmProjects/aiml_57/photo3.png' , img_convert)

#finding what colours are at what coordinates
px = img_colored[100,100]

#shape
print(img_colored.shape)

#size
print(img_colored.size)

#separating the image into its blue green and red components
photo4 = cv2.imread('dog.jpg',)
b,g,r = cv2.split(photo4)
zeros = np.zeros(b.shape ,np.uint8)

img_red = cv2.merge((zeros,zeros,r))
img_green = cv2.merge((zeros,g,zeros))
img_blue = cv2.merge((b,zeros,zeros))

cv2.imshow('red' , img_red)
cv2.waitKey(0)
cv2.imshow('green' , img_green)
cv2.waitKey(0)
cv2.imshow('blue' , img_blue)
cv2.waitKey(0)


#converting coloured image to gray image through cvt.color() function
photo5 = cv2.imread('dog.jpg')
converted_photo = cv2.cvtColor(photo5, cv2.COLOR_BGR2GRAY)
cv2.imshow('blackandwhite' , converted_photo)
cv2.waitKey(0)

#resizing an image
photo6 = cv2.imread('dog.jpg')
new_width = 300
new_height = 400
new_coordinates = (new_width,new_height)
resize = cv2.resize(photo6 , new_coordinates , interpolation=cv2.INTER_NEAREST)
cv2.imshow('resize' , resize)
cv2.waitKey(0)

#drawing a marker/rectangle on an image
photo7 = cv2.imread('dog.jpg')


start_point=(50,150)
end_point=(100,250)
color=(112,233,34)
thickness=4
rectangle=cv2.rectangle(photo7,start_point,end_point,color,thickness)

cv2.imshow('dog.jpg' , rectangle)
cv2.waitKey(0)

#rotating an image
rotate = cv2.imread('dog.jpg')

# dividing height and width by 2 to get the center of the image
height, width = rotate.shape[:2]
# get the center coordinates of the image to create the 2D rotation matrix
center = (width/2, height/2)

# using cv2.getRotationMatrix2D() to get the rotation matrix
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)

# rotate the image using cv2.warpAffine
rotated_image = cv2.warpAffine(src=rotate, M=rotate_matrix, dsize=(width , height))

cv2.imshow('rotate' , rotated_image)
cv2.waitKey(0)

#reading video
capture = cv2.VideoCapture(0) #the arguments will be 0,1,2,3 etc if we are using cameras connected to our pc
#0 : webcam
#1 : the first camera connected and so on....
#the argument will be a path of a video if we want to read a saved video.

while True :
    isTrue, frame = capture.read() # capture.frame() method reads the video frame by frame and
    # returns the frame andisTrue which is a boolean variable which tells whether the video has been read or
    # not.

    cv2.imshow('Video' , frame) # to display an indivisual frame of the video

    if cv2.waitKey(1) & 0xFF==ord('d') : #in this waitKey if i give 0 as a parameter then on every click
        #it takes a screenshot and displays that. writing a another number like 1 or 20 would display it
        #as a live video
        # 0xFF==ord('d') means to stop the video we need to press the letter d
        break # to stop the video from playing for ever.

capture.release() # releasing the capture variable
cv2.destroyAllWindows()




