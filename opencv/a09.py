import cv2
import numpy as np


input_img = cv2.imread("003.jpg")
img = cv2.resize(input_img, (640, 480))
input_image_cpy = img.copy()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#r254 g0 b0
lower_red = np.array([0, 250, 154])
upper_red = np.array([10, 300, 354])

mask_red = cv2.inRange(hsv, lower_red, upper_red)
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_red_cap = cv2.drawContours(input_image_cpy, contours_red, -1, (0, 0, 0), 3)
cv2.imshow('contour_red_cap', contour_red_cap)
cv2.waitKey(0)


