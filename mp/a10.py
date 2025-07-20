import cv2
import mediapipe as mp
import math

# 初始化 Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

def get_direction(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(-dy, dx))  # 上為正

    if -45 <= angle <= 45:
        return "RIGHT"
    elif 45 < angle <= 135:
        return "UP"
    elif angle > 135 or angle < -135:
        return "LEFT"
    elif -135 <= angle < -45:
        return "DOWN"
    else:
        return ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    direction = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            thumb_point = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_point = (int(index_tip.x * w), int(index_tip.y * h))

            # 畫線
            cv2.line(frame, thumb_point, index_point, (0, 255, 0), 3)
            cv2.circle(frame, thumb_point, 8, (0, 0, 255), -1)
            cv2.circle(frame, index_point, 8, (255, 0, 0), -1)

            # 計算方向
            direction = get_direction(thumb_point, index_point)

    # 顯示方向文字
    if direction:
        cv2.putText(frame, f"->: {direction}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

    cv2.imshow("sample", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 離開
        break

cap.release()
cv2.destroyAllWindows()
