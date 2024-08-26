# Football Analysis Project

## Overview
This project is designed to detect and track players, referees, and footballs within video footage using YOLO, a leading AI object detection model. The model is further trained to enhance its accuracy and performance. In addition to detection, players are assigned to teams based on their jersey colors using K-means clustering for pixel segmentation. This allows us to calculate team ball possession percentages during a match.

## The Tracking Mechanism
The tracking mechanism is the backbone of the model. It ensures that each detected object within the video is identified and assigned a unique identifier, maintaining this identity across each frame. The main components of the tracking mechanism are:

- **YOLO (You Only Look Once):** A powerful real-time object detection algorithm introduced in 2015, notable for its speed and versatility in detecting around 80 pre-trained classes. For this project, YOLOv8x, developed by Ultralytics, is used for its enhanced capabilities. YOLO can also be trained on custom datasets to detect specific objects.

- **ByteTrack Tracker:** Essential for Multiple Object Tracking (MOT), ByteTrack links detected objects across video frames, assigning unique track IDs to each object. ByteTrack was introduced in 2021 and is implemented using Python's supervision library in this project.

- **OpenCV:** A widely-used library for computer vision tasks, OpenCV is utilized here to visualize and annotate video frames with bounding boxes and text for each detected object.

To build the tracking mechanism, the following steps are initiated:

1. **Deploy YOLO with ByteTrack:** Detect objects (e.g., players) and assign unique track IDs.
2. **Initialize Object Tracks:** Store object tracks in a pickle (pkl) file to avoid re-executing the object detection process for each run, saving significant time.

## Screenshot
![Screenshot](/screen.png)

## Modules Used
This project leverages the following key modules:

- **YOLO:** A cutting-edge AI model for object detection.
- **K-means:** Used for pixel segmentation and clustering to identify jersey colors.
- **Optical Flow:** Employed to track camera movement across frames.
- **Perspective Transformation:** Converts scene depth and perspective into real-world metrics.
- **Player Metrics:** Speed and distance calculations per player.

## Trained Models
- YOLO v5 Model

## Sample Video
- Example of input video footage

## Requirements
To run this project, ensure you have the following installed:

- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib

## Acknowledgments

This project was inspired by the following video:

- [YouTube Video](https://www.youtube.com/watch?v=neBZ6huolkg&t=1s)
