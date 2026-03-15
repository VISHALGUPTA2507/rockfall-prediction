AI Rockfall Detection and Alert System
Overview

The AI Rockfall Detection and Alert System is a deep learning and computer vision based safety monitoring system designed to detect potential rockfall events in mining areas or mountainous regions.

The system uses PyTorch, OpenCV, and ResNet18 CNN to detect rockfall events from live camera feeds or video streams. When rock movement is detected, the system triggers alerts and saves snapshots for safety monitoring.

A web dashboard interface is also included to visualize monitoring data, weather conditions, mine blueprint, drone view, and CCTV monitoring.

Project Features
AI Rockfall Detection

Deep Learning model using ResNet18

Detects rockfall vs normal conditions

Trained using image dataset with two classes:

rockfall

normal

Real-Time Video Monitoring

Uses OpenCV for webcam or video stream input

Detects moving objects using background subtraction

Classifies moving regions using trained AI model

Alert System

When rockfall is detected:

Bounding box appears on detected area

Alert sound is triggered

Snapshot image is saved

Visual alert is displayed

Monitoring Dashboard

The system includes a web dashboard with:

Temperature monitoring

Weather information

Mine location

Mine blueprint view

Drone monitoring

CCTV monitoring

Notification alerts

The dashboard interface is implemented using HTML, CSS and JavaScript.

System Architecture
Camera / Video Feed
        |
        v
Motion Detection (OpenCV)
        |
        v
AI Model (ResNet18)
        |
        v
Rockfall Classification
        |
        v
Alert System + Dashboard Notification
Project Structure
rockfall-prediction/

dash_board3.html          # Web dashboard interface
project.py                # Main AI rockfall detection system
project_traindml.py       # Webcam detection script
rockfall_resnet18.pt      # Trained deep learning model

dataset/
    rockfall/
    normal/

alerts/
    saved_alert_images

README.md
Technologies Used
Programming Language

Python

Machine Learning

PyTorch

TorchVision

ResNet18 CNN

Computer Vision

OpenCV

Data Processing

NumPy

Pillow

Web Dashboard

HTML

CSS

JavaScript

Installation

Clone the repository:

git clone https://github.com/VISHALGUPTA2507/rockfall-prediction.git
cd rockfall-prediction

Install required libraries:

pip install torch torchvision opencv-python pillow numpy tqdm playsound
Dataset Structure

Your dataset should follow this structure:

dataset/

rockfall/
    img1.jpg
    img2.jpg
    img3.jpg

normal/
    img4.jpg
    img5.jpg
    img6.jpg

The system trains a classifier to distinguish rockfall images from normal images.

Training the Model

Run the training mode:

python project.py --mode train --data_dir dataset --epochs 10 --batch_size 16

The training script uses transfer learning with ResNet18 and saves the best model.

The trained model will be saved as:

models/rockfall_resnet18.pt
Running Real-Time Detection

Run rockfall detection using webcam:

python project.py --mode run --model_path models/rockfall_resnet18.pt --source 0

Run detection using a video file:

python project.py --mode run --model_path models/rockfall_resnet18.pt --source video.mp4

The system will:

Capture frames

Detect motion regions

Classify the region using AI model

Trigger alerts when rockfall probability is high.

Webcam Detection Script

You can also run a simple detection script:

python project_traindml.py

This script:

Opens webcam

Predicts rockfall or normal condition

Displays prediction on screen

Plays alert sound if rockfall detected.

Dashboard Interface

Open the dashboard in your browser:

dash_board3.html

Dashboard includes:

Environmental Monitoring

Temperature

Weather

GPS Location

Monitoring Views

Mine blueprint

Drone view

CCTV monitoring

Alert System

Rockfall alert button

Notification system

Real-time alerts

Alert Mechanism

When a rockfall event is detected:

Bounding box appears on detected region

Alert sound is triggered

Snapshot image is saved

Alert message is displayed
Snapshots are stored in:
alerts/
Future Improvements
Possible future upgrades:
IoT sensor integration
Real-time SMS alerts
Cloud monitoring system
Mobile application dashboard
GIS-based rockfall risk mapping
Improved deep learning models

Author
POTTA VISHALGUPTA
Skills:
Machine Learning
Data Science
Python
Deep Learning

Computer Vision
