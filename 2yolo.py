import os
import json
from pathlib import Path

im_width = 600
im_height = 600

LABEL_DICT = {
    "box": 0,
    "bottle": 1,
    "pouch": 2,
    "cylinder": 3,
    "book": 4
}

label_path = "CabinetData/labels/train"

# for f in os.listdir(label_path):
#     if f == ".DS_Store":
#         os.remove(f"{label_path}/{f}")
#         continue
#     if ".json" in f:
#         os.remove(f"{label_path}/{f}")
#         continue

for f in os.listdir(label_path):
    if f == ".DS_Store":
        os.remove(f"{label_path}/{f}")
        continue
    print(f)
    with open(f"{label_path}/{f}", "r", encoding="utf-8") as o:
        s = o.read()
    print(s)
    data = json.loads(s)

    with open(Path(f"{label_path}/{f}").with_suffix(".txt"), "a", encoding="utf-8") as file:
        for i in data['shapes']:
            label = i['label']
            start_point = i['points'][0]
            start_x = start_point[0]
            start_y = start_point[1]
            end_point = i['points'][1]
            end_x = end_point[0]
            end_y = end_point[1]
            width = abs(end_x - start_x)
            height = abs(end_y - start_y)
            mid_x = start_x + (width/2)
            mid_y = start_y + (height/2)
            mid_x_normalized = mid_x / im_width
            width_normalized = width / im_width
            mid_y_normalized = mid_y / im_height
            height_normalized = height / im_height
            line = str(LABEL_DICT[label]) + " " + str(mid_x_normalized) + " " + str(mid_y_normalized) + " " + str(width_normalized) + " " + str(height_normalized)
            file.write(line+"\n")