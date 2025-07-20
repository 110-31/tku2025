import cv2
import mediapipe as mp

# 初始化 Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

# 狀態控制
state = "idle"           # 狀態：idle → wait_4 → ok
index_state = "down"
index_flag = False       # 食指 是否完成
multi_state = "down"
multi_flag = False       # 4 指是否完成
ok_display_counter = 0   # 顯示 OK 

# 判斷是否手指伸直
def is_finger_up(landmarks, tip, pip):
    return landmarks[tip][1] < landmarks[pip][1]

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

            # 抓 landmark
            landmarks = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks.landmark]

            # 各指判斷是否伸直
            index_up = is_finger_up(landmarks, 8, 6)
            middle_up = is_finger_up(landmarks, 12, 10)
            ring_up = is_finger_up(landmarks, 16, 14)
            pinky_up = is_finger_up(landmarks, 20, 18)

            # ========== 第一階段：偵測 index 單指 ==========
            if state == "idle":
                if index_up and not any([middle_up, ring_up, pinky_up]):
                    if index_state == "down":
                        index_state = "up"
                elif not index_up:
                    if index_state == "up":
                        index_state = "down"
                        state = "wait_4"
                        print("完成 食指 1 → 進入 wait_4")

            # ========== 第二階段：偵測 4 指（index, middle, ring, pinky） ==========
            elif state == "wait_4":
                # 若 4 指都伸直
                if index_up and middle_up and ring_up and pinky_up:
                    if multi_state == "down":
                        multi_state = "up"
                elif not any([index_up, middle_up, ring_up, pinky_up]):
                    if multi_state == "up":
                        multi_state = "down"
                        state = "ok"
                        ok_display_counter = 30
                        print("完成 ➜  OK! OK!")
                else:
                    # 偵測到不是 4 指或其他亂動，視為中斷，重置
                    state = "idle"
                    index_state = "down"
                    multi_state = "down"
                    print("  NO  手勢錯誤或中斷，重置")

    # 畫面狀態顯示
    cv2.putText(frame, f"State: {state}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if ok_display_counter > 0:
        cv2.putText(frame, "OK", (120, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        ok_display_counter -= 1
        if ok_display_counter == 0:
            state = "idle"

    cv2.imshow("sample", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
