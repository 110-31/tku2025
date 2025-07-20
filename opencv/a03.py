import cv2

# 載入 Haar Cascade 模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 開啟攝影機
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 偵測人臉
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # 繪製人臉矩形（可選）
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray  = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # 偵測眼睛
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # 計算眼睛中心點
            center_x = ex + ew // 2
            center_y = ey + eh // 2
            radius = int(round((ew + eh) * 0.25))  # 半徑可根據實際調整
            # 畫圓框標記眼睛
            cv2.circle(roi_color, (center_x, center_y), radius, (0, 255, 0), 2)

    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
