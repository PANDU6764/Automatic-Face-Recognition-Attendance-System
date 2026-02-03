 # 📸 Real-Time Face Recognition Attendance System
 
---

A real-time **face recognition–based attendance system** that automatically marks attendance using a live camera feed.  
The system runs for a **fixed 30-second session**, prevents duplicate attendance, and stores records locally with a strong focus on **privacy and reliability**.

---

## Overview

This work implements an automated attendance workflow using **computer vision and deep learning**.  
It uses a **pretrained FaceNet model** for face recognition and applies **temporal identity validation** to ensure that each individual is marked **only once per session**.

All sensitive data such as face images, databases, and attendance logs are excluded from version control.

---

## Key Features

- Real-time face detection using MTCNN  
- Face recognition using pretrained FaceNet (Keras)  
- Automatic 30-second attendance session  
- On-screen countdown timer  
- Duplicate and false attendance prevention  
- Laptop camera support  
- Privacy-safe local data storage  

---

## Project Structure

<p align="center"><i>Project structure of the Automatic Face Recognition Attendance System</i></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/PANDU6764/Automatic-Face-Recognition-Attendance-System/main/project%20structure.png" width="600"/>
</p>

- Dataset placement: data/enrollment/
- Attendance output should visible: data/snapshots/attendance.xlsx
- Attendance output images should visible: data/snapshots/images

> Note: Real-time datasets, databases, and attendance files are intentionally excluded for privacy reasons.
---

## Technology Stack

- Python
- OpenCV
- TensorFlow/Keras
- FaceNet
- MTCNN
- SQLite
- OpenPyXL 

---

## Setup Instructions

### Pretrained Model Setup

This work uses a **pretrained FaceNet Keras model**.

You can download it from:

- 🔗 FaceNet Keras model on Kaggle:

  https://www.kaggle.com/datasets/suicaokhoailang/facenet-keras
  
After downloading, place it here:
          
--> models/facenet_keras.h5

### Clone the Repository
```bash
git clone https://github.com/PANDU6764/Automatic-Face-Recognition-Attendance-System.git
cd Attendance
pip install -r requirements.txt
python app/run.py
```
## How It Works

- Faces are detected in each frame using MTCNN
- FaceNet converts faces into embeddings
- Embeddings are matched using L2 distance
- Identity is confirmed across multiple frames
- Attendance is recorded once per individual

<p align="center">
  <img src="https://raw.githubusercontent.com/PANDU6764/Automatic-Face-Recognition-Attendance-System/main/camera_detection.png" width="600"/>
</p>

<p align="center"><i>Real-time face detection during attendance session</i></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/PANDU6764/Automatic-Face-Recognition-Attendance-System/main/Attendance.png" width="600"/>
</p>



  

  



