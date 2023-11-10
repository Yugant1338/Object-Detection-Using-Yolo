
from ultralytics import YOLO
import cv2
import scipy
import cvzone
import math
import numpy

cap = cv2.VideoCapture(0)
#cv2.VideoCapture(0) integrates the compiler with the webcam present in the pc(if any) and then it starts the detection in real time frame
classname=['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
cap.set(3,800)
cap.set(4,600)
# video is being resized to 800x600 dimesion
model = YOLO('yolo weights/yolov8l.pt')
while True:
    # while the user provides with an interuppt the code with run and detect the objects in the video frame per frame on a loop
    success, img=cap.read()
    success, img=cap.read()
    results=model(img, stream=True)
    for r in results:
        boxes=r.boxes

        kpt=r.keypoints
        if kpt is not None:
            kpt = kpt.cpu().numpy()
            print(kpt)
        else:
            kpt = []
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            cvzone.putTextRect(img,f'{classname[cls]} {int(conf*100)}%',(max(0,x1),max(35,y1-20)))
            # cvzone.cornerRect(img,(x1,y1,x2,y2))

    cv2.imshow("Image",img)
    cv2.waitKey(1)





