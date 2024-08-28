# Soccer Game Footage Analysis

## Description
To analyze soccer game footage, we train a custom ML model on a public dataset of Bundesliga soccer game footage. Key features of the project include:
- Tracking the ball and players frame-by-frame in the video and adjusting for camera movement
- Object detection to identify the players, referees, and ball
- Assigning players to teams based on jersey color
- Assigning possession of the ball to player based on proximity and calculating team possession statistics
- Calculating player speed and distance traveled statistics from frame data

## Technologies
The following modules are used in this project:
- YOLO: AI object detection model
- Optical Flow: Measure camera movement
- KMeans: Pixel segmentation and clustering to detect t-shirt color
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Sample video input and output
- Input: https://drive.google.com/file/d/1L39lM-dQ1tVHc4066bRoXhIIPIIPmZt3/view?usp=drive_link
- Output: https://drive.google.com/file/d/1EDCCkad6-CiFRDSPnTzP5Z7VwBRrC5pp/view?usp=drive_link

## Test the project
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

Steps:
1. Train your own model based on the sample notebook code in the /training folder and add it to the project
2. Modify the paths specified in main.py if needed to match the file location of your model and an input video of your choice
3. Run main.py and view the analyzed video in the /output_videos folder
