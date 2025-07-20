import cv2
import time
import threading
import numpy as np
from flask import Flask, render_template, Response

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize

app = Flask(__name__)

# 模型路徑
model_path = "rock-paper-scissors.tflite"

# 儲存辨識結果與鎖
detection_result_list = []
lock = threading.Lock()


def detect_objects():
    # 啟動 webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    counter, fps = 0, 0
    fps_avg_frame_count = 10
    start_time = time.time()

    # 定義 callback 給 LIVE_STREAM 模式
    def callback(result: vision.ObjectDetectorResult, output_image: mp.Image, timestamp_ms: int):
        with lock:
            result.timestamp_ms = timestamp_ms
            detection_result_list.clear()
            detection_result_list.append((result, output_image.numpy_view()))

    # 建立 ObjectDetector（包含 callback）
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        score_threshold=0.5,
        result_callback=callback
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # 持續讀取影像進行偵測
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        counter += 1
        detector.detect_async(mp_image, counter)

        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()


def generate_frames():
    while True:
        with lock:
            if detection_result_list:
                result, img_array = detection_result_list[0]
                vis_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                vis_img = visualize(vis_img, result)
            else:
                continue

        _, buffer = cv2.imencode('.jpg', vis_img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    thread = threading.Thread(target=detect_objects)
    thread.daemon = True
    thread.start()
    app.run()
