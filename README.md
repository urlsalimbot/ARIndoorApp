# Vision-based machine learning indoor AR navigation and positioning project(Python-Based)

This project implements an indoor AR navigation system using computer vision and machine learning techniques. It features real-time SLAM (Simultaneous Localization and Mapping), AR visualization, and camera calibration capabilities.

## Requirements

- Python 3.10 or 3.11 (Python 3.12 is not yet fully supported)
- A webcam or camera device
- Windows, Linux, or macOS

## Installation

1. Install Python 3.11 from [python.org](https://www.python.org/downloads/)

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Features

- Real-time camera feed with AR overlay
- Feature-based SLAM using ORB features
- Kalman filter-based pose estimation
- Camera calibration using chessboard pattern
- 3D navigation path visualization
- Waypoint marking and visualization

## Usage

1. Run the application:
```bash
python src/main.py
```

2. Camera Calibration:
   - Print a 9x6 chessboard pattern
   - Click the "Calibrate" button
   - Show the chessboard pattern to the camera from different angles
   - After capturing 10 different views, click "Save Calibration"
   - The calibration parameters will be saved for future use

3. Navigation:
   - Click "Start Navigation" to begin AR navigation
   - The app will track your position using SLAM
   - AR overlay will show navigation path and waypoints
   - Click "Stop" to pause navigation

## Technical Details

### Mobile App (Python-Based)

- Kivy/KivyMD → Best for cross-platform mobile UI
- PyOpenGL with Kivy → For real-time 3D rendering on mobile
- PySide/PyQt (Alternative if desktop UI is also needed)

### Feature-Based SLAM (On-Device)

- ORB-SLAM3 → Optimized for mobile (lightweight and real-time)
- SuperGlue + SuperPoint (ONNX/TFLite) → Deep learning-based feature matching

### IMU Sensor Fusion (For More Accuracy)

- filterpy → Kalman Filter
- Custom pose graph optimization

### Cloud-Based ML Processing (Offloaded Computation)

- FastAPI → Lightweight Python-based API for real-time communication
- TensorFlow Serving / ONNX Runtime Server → Deploy pre-trained ML models for:
  - Depth Estimation (MiDaS, DPT)
  - Object Detection (YOLOv8-Nano)
  - Scene Reconstruction (COLMAP)
- MQTT/WebSockets → For real-time data transfer between mobile and server
- PostgreSQL/PostGIS → Storing 3D maps and spatial data

### On mobile (Real-time Rendering)

- PyOpenGL + Kivy → 3D rendering inside the mobile app
- pythreejs → If using Web-based 3D visualization
- Open3D → Point cloud visualization
- VisPy → Real-time graphics rendering

### On Server (For More Complex 3D Mapping)

- Blender-Python → If detailed 3D environment models are needed
- COLMAP → 3D scene reconstruction from images
- PCL (Point Cloud Library) → Processing LiDAR/Depth data

## Troubleshooting

If you encounter installation issues:
1. Make sure you're using Python 3.10 or 3.11
2. On Windows, you might need to install Visual C++ Build Tools
3. Some packages might require additional system dependencies:
   - On Windows: Install Visual C++ Redistributable
   - On Linux: Install OpenGL, SDL2, and GStreamer development packages
   - On macOS: Install XCode Command Line Tools

## Contributing

Feel free to submit issues and enhancement requests!
