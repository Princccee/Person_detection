import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from Person_tracker import *
import os
from Best_shot import *
from Best_shot import choose_best_image

# Provide the path to video here:
cap = cv2.VideoCapture('Laundry_guy.mp4')

model = YOLO('../Yolo-weights/yolov8n.pt')  # Nano model
# model = YOLO("../Yolo-Weights/yolov8l.pt") # Large model

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]

snapshot_folder = 'snapshots'

# if a snapshot folder doesn't exist then create one:
if not os.path.exists(snapshot_folder):
    os.makedirs(snapshot_folder)

snapshot_counter = 0
detected_objects = set()  # To keep track of detected objects

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box(Using CV2):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Bounding Box(Using Cvzone):
            w, h = (x2 - x1), (y2 - y1)

            # Confidence level:
            conf = math.ceil(box.conf[0] * 100) / 100

            # Class Name:
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'person' and conf > 0.9 and (x1, y1, x2, y2) not in detected_objects:
                cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5)

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

                # Save snapshot
                snapshot_path = os.path.join(snapshot_folder, f'snapshot_{snapshot_counter}.jpg')
                cv2.imwrite(snapshot_path, img)
                snapshot_counter += 1

                # Add detected object to set
                detected_objects.add((x1, y1, x2, y2))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        print(result)
        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(0, y1 - 20)),
                           scale=2, thickness=3, offset=10)

    try:
        cv2.imshow("Image", img)
    except:
        break
    cv2.waitKey(1)

# After the loop ends, select the best image and delete the snapshot folder
print("hello")
best_image = choose_best_image(snapshot_folder)
cv2.imshow("Best Image", best_image)
cv2.waitKey(0)
