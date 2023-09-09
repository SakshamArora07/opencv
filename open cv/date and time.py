# show date and time on a live video
# shapes and text can be put on a video as well in a similar way

import cv2
import datetime

capture = cv2.VideoCapture(0)

while True :
    isTrue, frame = capture.read()


    cv2.putText(frame ,'date and time : ' + str(datetime.datetime.now()) , (10,50) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
    cv2.imshow('Video' , frame)

    if cv2.waitKey(1) & 0xFF==ord('q') :
      break

capture.release()
cv2.destroyAllWindows()

