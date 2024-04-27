# imagine2024-vision

## Description

This repository holds code to run a simple YOLOv5 loop for CSH's BitsNBytes project with a specified camera, video, or image input to detect relevant objects and extract regions of interest from specified zones. 

Most code in this repository is hard-coded for specific circumstances, but could be modified to fit different circumstances or made more scalable for different use cases. 

## Getting Started

To utilize our code, enter the following commands.

```shell
git clone https://github.com/ComputerScienceHouse/imagine2024-vision.git # clone this repo
pip install -r requirements.txt # install requirements
git clone https://github.com/ultralytics/yolov5.git # clone yolov5 into this directory
cd yolov5 
pip install -r requirements.txt # install requirements for yolov5
cd .. # return to main directory
```

For more instructions about how to utilize YOLOv5, see: https://github.com/ultralytics/yolov5

To run the main vision loop, simply run `main.py`

```shell
python main.py
```

### Points of Interest:
- Line 39: adjust this to change the yolo model to be used for inference
- Line 45: adjust this to change coordinates for desired polygon zone to extra regions of interest from
  - Run `getcoords.py` on a screenshot of your desires video or camera to determine which points you want for your polygon zone
- Line 48: change this according to your desired fps if you want to utilize supervision's tracker
- Line 56: change "cabinetview.webm" to your desired video or camera
- Line 62: change this according to your fps and resolution

### Utility Files:
- `2yolo.py`
  - Converts json output from anylabeling to format needed for YOLO training.
  - See: https://github.com/vietanhdev/anylabeling
- `getcoords.py`
  - Allows you to determine points of interest for creating polygon zones with supervision
- `resize_data.py`
  - Self-explanatory: given YOLO training data, resizes all images to desired resolution
- `verify-labels.py`
  - Denormalizes labels from a YOLO training set and creates a folder of all images with annotations to give visual confirmation of dataset accuracy 
