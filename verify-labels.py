import cv2
import supervision as sv
import os
import numpy as np
from pathlib import Path

img_path = "CabinetData/images/train"
label_path = "CabinetData/labels/train"
annotated_path = "CabinetData/annotated"

box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


os.makedirs(annotated_path, exist_ok=True)
for f in os.listdir(img_path):
    if f == ".DS_Store" or ".json" in f:
        os.remove(f"{img_path}/{f}")
        continue
    print(f"{img_path}/{f}")
    frame = cv2.imread(f"{img_path}/{f}")

    w = frame.shape[1]
    h = frame.shape[0]
    print(w, h)
    xyxy = []
    class_id = []
    l = Path(f"{label_path}/{f}").with_suffix(".txt")
    with open(l, "r") as o:
        for line in o.readlines():
            c, xc, yc, wi, hi = line.strip().split()

            print(c, xc, yc, wi, hi)
            x1 = float(xc) * w - ((float(wi)*w) / 2)
            y1 = float(yc) * h + ((float(hi)*h) / 2)
            x2 = float(xc) * w + ((float(wi)*w) / 2)
            y2 = float(yc) * h - ((float(hi)*h) / 2)
            print(c, x1, y1, x2, y2)

            xyxy.append([float(x1), float(y1), float(x2), float(y2)])
            class_id.append(int(c))
            print(xyxy, class_id)
    
    detections = sv.Detections(xyxy=np.array(xyxy), class_id=np.array(class_id))
    
    frame = box_annotator.annotate(frame, detections)
    frame = label_annotator.annotate(frame, detections)
    cv2.imwrite(f"{annotated_path}/{f}", frame)
print("Output saved to ", annotated_path)
