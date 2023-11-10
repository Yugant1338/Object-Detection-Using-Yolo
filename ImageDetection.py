
from ultralytics import YOLO
import cv2
import scipy
import cvzone
import math
import numpy

img=cv2.imread(r"C:\Users\raajc\PycharmProjects\ObjectDetection\venv\360_F_388979227_lKgqMJPO5ExItAuN4tuwyPeiknwrR7t2.jpg")
classname=['person','bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
img=cv2.resize(img,(800,600))

model = YOLO('yolo weights/yolov8l.pt')

results=model(img, stream=True)
for r in results:
    boxes=r.boxes

    kpt=r.keypoints
if kpt is not None:
            kpt = kpt.cpu().numpy()
            print(kpt)
else:
        kpt = []
        print(kpt)
for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 200), 1)

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h))

        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])
        cvzone.putTextRect(img, f'{classname[cls]} {int(conf * 100)}%', (max(0, x1), max(35, y1 - 20)))
        # cvzone.cornerRect(img,(x1,y1,x2,y2))

cv2.imshow("Image",img)
cv2.waitKey(0)





