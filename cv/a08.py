import cv2
import numpy as np


input_img = cv2.imread("002.png")
img = cv2.resize(input_img, (1464, 559))

#B-215 G-215 R-248
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#BGR
lower_red = np.array([148, 115, 115])
upper_red = np.array([348, 315, 315]) 

mask_red = cv2.inRange(hsv, lower_red, upper_red) 

cv2.imshow('mask_red', mask_red)
cv2.waitKey(0)

