import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

def is_finger_folded(landmarks, tip, pip):
    return landmarks[tip][1] > landmarks[pip][1]

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

            # 把 landmarks 存成 (x, y)
            landmarks = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks.landmark]

            # 檢查四指是否縮起來
            folded_fingers = 0
            finger_pairs = [(8,6), (12,10), (16,14), (20,18)]
            for tip, pip in finger_pairs:
                if is_finger_folded(landmarks, tip, pip):
                    folded_fingers += 1

            # 當 4 指都縮起來才繼續判斷拇指方向
            if folded_fingers == 4:
                thumb_tip = landmarks[4]
                thumb_mcp = landmarks[2]
                thumb_ip = landmarks[3]

                # 拇指向上
                if thumb_tip[1] < thumb_mcp[1]:
                    gesture = "GOOD"
                    color = (0, 255, 0)
                # 拇指向下
                elif thumb_tip[1] > thumb_mcp[1]:
                    gesture = "BAD"
                    color = (0, 0, 255)
                else:
                    gesture = ""
                    color = (255, 255, 255)

                if gesture:
                    cv2.putText(frame, gesture, (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

    cv2.imshow('sample', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
