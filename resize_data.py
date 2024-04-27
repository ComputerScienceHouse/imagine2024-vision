import cv2
import os

img_path = "GroceryInContext/images/train"
label_path = "GroceryInContext/labels/train"
resized_path = "GroceryInContext/resized"

os.makedirs(resized_path, exist_ok=True)

for f in os.listdir(img_path):
    if f == ".DS_Store":
        os.remove(f"{img_path}/{f}")
        continue
    print(f"{img_path}/{f}")

    img = cv2.imread(f"{img_path}/{f}")
    img = cv2.resize(img, (600, 600))
    cv2.imwrite(f"{resized_path}/{f}", img)
print("Done")