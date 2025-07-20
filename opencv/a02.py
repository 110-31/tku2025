import cv2
img = cv2.imread('face03.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 圖片轉灰階
#gray = cv2.medianBlur(gray, 5)                # 如果一直偵測到雜訊，可使用模糊的方式去除雜訊
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
nose_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')


# 偵測人臉
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
for (x, y, w, h) in faces:
    # 繪製人臉矩形（可選）
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_gray  = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # 偵測眼睛
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        # 計算眼睛中心點
        center_x = ex + ew // 2
        center_y = ey + eh // 2
        radius = int(round((ew + eh) * 0.25))  # 半徑可根據實際調整
        # 畫圓框標記眼睛
        cv2.circle(roi_color, (center_x, center_y), radius, (0, 255, 0), 2)


    nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")    # 使用鼻子模型
    noses = nose_cascade.detectMultiScale(gray)                             # 偵測鼻子
    for (x, y, w, h) in noses:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)  

    # mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml") # 使用嘴巴模型
    # mouths = mouth_cascade.detectMultiScale(gray)
    # for (x, y, w, h) in mouths:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)


cv2.imshow('image', img)
cv2.waitKey(0)   # 按下任意鍵停止
cv2.destroyAllWindows()