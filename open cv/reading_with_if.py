import cv2

img = cv2.imread('./photos/dog.jpg')
cv2.imshow('image' , img)

k = cv2.waitKey(0) # & 0xFF - mask is optional

if k == 27 : #escape button
    cv2.destroyAllWindows()
elif k == ord('s') :
    cv2.imwrite('if_elif_image.jpg' , img)
    cv2.destroyAllWindows()