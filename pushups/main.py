import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from pathlib import Path
import cv2
import time
import numpy as np

def process(image, keypoints):
    nose_visible = keypoints[0][0] > 0 and keypoints[0][1] > 0
    left_ear_visible = keypoints[3][0] > 0 and keypoints[3][1] > 0
    right_ear_visible = keypoints[4][0] > 0 and keypoints[4][1] > 0

    l_shldr = keypoints[5]
    r_shldr = keypoints[6]
    r_elbw = keypoints[7]
    l_elbw = keypoints[8]
    r_hand = keypoints[9]
    l_hand = keypoints[10]

    try:
        if left_ear_visible and not right_ear_visible:
            elbow_angle = angle(l_shldr, l_elbw, l_hand)
        else:
            elbow_angle = angle(r_shldr, r_elbw, r_hand)

        x, y = int(l_elbw[0]), int(l_elbw[1])
        cv2.putText(image, f"{int(elbow_angle)}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 25, 255), 2)

        return elbow_angle

    except ZeroDivisionError:
        pass

    return None


def angle(a, b, c):
    d = np.arctan2(c[1] - b[1], c[0] - b[0])
    e = np.arctan2(a[1] - b[1], a[0] - b[0])
    angle_ = np.rad2deg(d - e)
    angle_ = angle_ + 360 if angle_ < 0 else angle_
    return 360 - angle_ if angle_ > 180 else angle_


model_path = "yolo11n-pose.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

prev_time = time.time()
down_phase = False
pushup_count = 0
writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"avc1"), 10, (640, 480))

angle_history = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    writer.write(frame)
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 255, 25), 1)
    cv2.imshow('YOLO', frame)

    results = model(frame)

    if cv2.waitKey(1) == ord("q"):
        break

    if not results:
        continue

    result = results[0]
    keypoints_data = result.keypoints.xy.tolist()
    if not keypoints_data:
        continue

    keypoints = keypoints_data[0]
    if not keypoints:
        continue

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated_frame = annotator.result()

    current_angle = process(annotated_frame, keypoints)

    if current_angle is not None:
        angle_history.append(current_angle)
        if len(angle_history) > 5:
            angle_history = angle_history[-5:]  

        avg_angle = np.mean(angle_history)

        if avg_angle < 100:
            down_phase = True
        elif down_phase and avg_angle >= 100:
            pushup_count += 1
            down_phase = False

    cv2.putText(annotated_frame, f"Total: {pushup_count}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 255, 25), 1)
    cv2.imshow("Pose", annotated_frame)

writer.release()
cap.release()
cv2.destroyAllWindows()
