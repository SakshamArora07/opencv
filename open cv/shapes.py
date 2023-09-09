# drawing different geometric shapes and text on an image
import cv2

img = cv2.imread('./photos/dog.jpg')

# line --- img = cv2.line(img , coordinate 1 in the form of tuple , coordinate 2 in the form of tuple , color in BGR format , thickness in number )
img = cv2.line(img , (0,0) , (255,255) , (0,0,255),2 )

# arrowed line --- img = cv2.line(img , coordinate 1 in the form of tuple , coordinate 2 in the form of tuple , color in BGR format , thickness in number )
img = cv2.arrowedLine(img , (10,0) , (104,24) , (0,0,255),2)

#rectangle
# img = cv2.rectangle(img , top left vertex coordinats , bottom right coordinates , color in bgr format, thickness / -1 : -1 means it fills the rectangle with the mentioned color)
img = cv2.rectangle(img , (30,130) , (200,80) , (200,62,200), -1)
img = cv2.rectangle(img , (30,130) , (200,20) , (100,62,100), 2)

#circle
# img = cv2.circle(img , centre of the circle , radius of the circle, color in bgr format, thickness / -1 : -1 means it fills the rectangle with the mentioned color)
img = cv2.circle(img , (100,100) , 50 , (200,0,0) , -1)
img = cv2.circle(img , (190,100) , 50 , (200,0,0) , 3)

#text
# img = cv2.putText(img , 'text you want to put' , 'starting point of the text , font face/type , font size , color of the font in bgr , thickness , line type)
img = cv2.putText(img,'hi',(25,100),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),5,cv2.LINE_AA)

#similarly there are methods for polygon : cv2.polyLine , ecllipse : cv2.ecllipse etc

cv2.imshow('shapes', img)

cv2.waitKey(0)
cv2.destroyAllWindows()