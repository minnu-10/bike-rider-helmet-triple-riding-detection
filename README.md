# Detection of Bike Riders Without Helmet and Triple Riding Using Machine Learning and Web Surveillance

## Overview

This project is a real-time machine learning-based traffic surveillance system designed to detect bike riders who are:

- Riding without a helmet
- Involved in triple riding
- Involved in both triple riding and driver helmet violation

The system uses a live camera stream from a smartphone/IP webcam and combines object detection, rider classification, helmet detection, and rule-based violation analysis.

When a violation is confirmed, the system automatically generates an evidence image and a fine report.

---

## Features

- Real-time bike and rider detection
- Helmet detection using YOLOv3
- Person and vehicle detection using YOLOv8
- Detection of motorcycles and bicycles
- Pedestrian identification and filtering
- Child detection and exemption
- Triple-riding detection
- Driver identification based on rider position
- Temporal smoothing to reduce false detections
- Adaptive frame skipping for performance optimization
- GPU acceleration using CUDA when available
- Automatic violation evidence generation
- Automatic fine report generation
- Real-time monitoring interface

---

## System Architecture

```text
                    Live Camera / IP Webcam
                             |
                             v
                    Video Frame Capture
                             |
                             v
                    YOLOv8 Person & Vehicle
                         Detection
                             |
              +--------------+--------------+
              |                             |
              v                             v
        Person Detection              Vehicle Detection
              |                             |
              v                             v
       Pedestrian Filtering        Rider Association
              |                             |
              |                             v
              |                    Adult / Child
              |                     Classification
              |                             |
              |                             v
              |                      Driver Detection
              |                             |
              |                             v
              |                     Helmet Detection
              |                             |
              +--------------+--------------+
                             |
                             v
                    Violation Rule Engine
                             |
              +--------------+--------------+
              |              |               |
              v              v               v
          No Helmet     Triple Riding    Combined
              |              |               |
              +--------------+---------------+
                             |
                             v
                  Temporal Confirmation
                             |
                             v
                  Violation Confirmed
                             |
              +--------------+--------------+
              |                             |
              v                             v
        Evidence Image                Fine Report
