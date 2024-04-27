import cv2
import torch
import numpy as np
import supervision as sv
import os
from pathlib import Path


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, exist_ok=False,
    sep="", mkdir=False.

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #ss
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', force_reload=True)  # local repo
#model = torch.hub.load('yolov5', 'yolov5l', source='local', force_reload=True)

save_dir = increment_path(Path(os.getcwd()) / 'runs/exp')
save_dir.mkdir(parents=True, exist_ok=True)

polygon = np.array([[363, 598], [367, 0], [800, 0], [800, 598]])


tracker = sv.ByteTrack(frame_rate=20)

box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

zone = sv.PolygonZone(polygon)
zone_annotator = sv.PolygonZoneAnnotator(zone, color=sv.Color.RED)

cap = cv2.VideoCapture("cabinetview.webm")
fps = cap.get(cv2.CAP_PROP_FPS)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter.fourcc(*'mp4v')

out = cv2.VideoWriter(filename= f"{save_dir}/output.mp4", fourcc= fourcc, fps=20, frameSize=(800, 600))

CLASS_NAMES_DICT = model.model.names

with torch.inference_mode():
    curr_frame = 0
    try:
        while True:
            ret, frame = cap.read()

            if frame is None:
                break
            curr_frame+=1

            result = model(frame)
            print(result)
            detections = sv.Detections.from_yolov5(result)
            #detections = tracker.update_with_detections(detections)
            

            frame = box_annotator.annotate(frame, detections)
            frame = label_annotator.annotate(frame, detections)
            

            in_zone = zone.trigger(detections)
            #frame = zone_annotator.annotate(frame)
            
            for i in range(len(in_zone)):
                if in_zone[i]:
                    x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                    class_id = detections.class_id[i]
                    name = CLASS_NAMES_DICT[class_id]
                    try:
                        width = x2 - x1
                        height = y2 - y1
                        # Region of Image (ROI), where we want to save
                        roi = frame[y1:y2, x1:x2]
                        t = round(curr_frame/20, 2)
                        
                        cv2.imwrite(f"{save_dir}/object-at-{str(t).replace('.', '-')}s.jpg", roi)
                        
                    except Exception as e:
                        if type(e) == IndexError:
                            pass  # occurs when bounding box is outside of the frame
                        else:
                            print(e)
                    # call algorithm

            out.write(frame)
            cv2.imshow("BitsNBytes", frame)

            if (cv2.waitKey(30) == 27):
                break
            if cv2.waitKey(12) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)
    finally:
        print("Recording quit.")
        print(CLASS_NAMES_DICT)
        cap.release()
        out.release()
        cv2.destroyAllWindows()