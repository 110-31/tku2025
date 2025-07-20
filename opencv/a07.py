import cv2


input_img = cv2.imread("002.png")
img = cv2.resize(input_img, (1464, 559))
input_image_cpy = img.copy() 
cv2.imshow('image', img)
cv2.waitKey(0)