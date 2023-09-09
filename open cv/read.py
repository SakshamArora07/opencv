import cv2
import cv2 as cv

#reading image
img = cv.imread('./photos/dog.jpg' )
#img = cv.imread('./photos/dog.jpg' , 0 / 1 / -1 )
# 1 : loads a color image (Default)
# 0 : loads image in grayscale mode
# -1 : loads image as it is including alpha channel

#function to display the image in a new window :
cv.imshow('Dog' , img) #the first parameter is the name of the window and the second parameter
# is the name of the variable in which the image is stored.

cv.waitKey(0) #waits for a key to be pressed and 0 means it waits for infinite amount of time for a key to be pressed.
# the argument is the amount of time for which it waits for the image to disappear.

cv.destroyAllWindows()

# writing an image / saving an image
cv.imwrite('saved_pic.jpg' , img)
#first argument is the name of the file after saving and second argument is the variable in which the image is stored.

#reading video
capture = cv.VideoCapture(0) #the arguments will be 0,1,2,3 etc if we are using cameras connected to our pc
#0 : webcam
#1 : the first camera connected and so on....
#the argument will be a path of a video if we want to read a saved video.

while True :
    isTrue, frame = capture.read() # capture.frame() method reads the video frame by frame and
    # returns the frame and isTrue which is a boolean variable which tells whether the video has been read or not.

    cv.imshow('Video' , frame) # to display an indivisual frame of the video

    if cv.waitKey(1) & 0xFF==ord('q') : #in this waitKey if w give 0 as a parameter then on every click
        #it takes a screenshot and displays that. writing a another number like 1 or 20 would display it
        #as a live video
        # 0xFF==ord('q') means to stop the video we need to press the letter q
        break # to stop the video from playing for ever.

capture.release() # releasing the capture variable
cv.destroyAllWindows()

# after the video gets completed it throws an error which means that
#opencv could not find a video/photo at the path mentioned now since the video got over
#but it is fine let that error come.


#reading video in grayscale form
capture = cv.VideoCapture(0) #the arguments will be 0,1,2,3 etc if we are using cameras connected to our pc
#0 : webcam
#1 : the first camera connected and so on....
#the argument will be a path of a video if we want to read a saved video.

while True :
    isTrue, frame = capture.read() # capture.frame() method reads the video frame by frame and
    # returns the frame and isTrue which is a boolean variable which tells whether the video has been read or not.

    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    cv.imshow('Video' , gray) # to display an indivisual frame of the video

    if cv.waitKey(1) & 0xFF==ord('q') : #in this waitKey if w give 0 as a parameter then on every click
        #it takes a screenshot and displays that. writing a another number like 1 or 20 would display it
        #as a live video
        # 0xFF==ord('q') means to stop the video we need to press the letter q
        break # to stop the video from playing for ever.

capture.release() # releasing the capture variable
cv.destroyAllWindows()

# video properties
capture = cv.VideoCapture(0)

# for capturing/saving a video
fourcc = cv2.VideoWriter_fourcc( * 'XVID')
out = cv2.VideoWriter('output.avi' , fourcc , 20.0 , (640,480))
#20 is the no of frames per second we want to capture
#(640,480) is the tuple which has the frame size which we want to capture

while True :
    isTrue, frame = capture.read()

    print(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) # frame width
    print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # frame height

    out.write(frame) # for capturing/saving video

    cv.imshow('Video' , frame)

    if cv.waitKey(1) & 0xFF==ord('q') :
        break

capture.release()
cv.destroyAllWindows()






