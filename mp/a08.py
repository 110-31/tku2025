import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

# 初始化狀態
prev_index_up = False
index_up_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 抓 landmark 座標
            landmarks = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks.landmark]

            # 判斷食指是否朝上
            tip_y = landmarks[8][1]
            pip_y = landmarks[6][1]

            is_index_up = tip_y < pip_y  # 是否伸直

            # 偵測狀態從 Down -> Up（代表一次新伸直）
            if is_index_up and not prev_index_up:
                index_up_count += 1

            prev_index_up = is_index_up  # 更新狀態

    # 顯示計數結果
    cv2.putText(frame, f'Index Up Count: {index_up_count}', (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow('sample', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
