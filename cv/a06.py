
import cv2

img = cv2.imread('001.jpg')
cv2.imshow('image', img)
# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()


