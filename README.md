# Vision-based machine learning indoor AR navigation and positioning project(Python-Based)

Mobile App (Python-Based)

- Kivy/KivyMD → Best for cross-platform mobile UI
- PyOpenGL with Kivy → For real-time 3D rendering on mobile
- PySide/PyQt (Alternative if desktop UI is also needed)

Feature-Based SLAM (On-Device)

- ORB-SLAM3 → Optimized for mobile (lightweight and real-time)
- SuperGlue + SuperPoint (ONNX/TFLite) → Deep learning-based feature matching

IMU Sensor Fusion (For More Accuracy)

- filterpy → Kalman Filter
- GTSAM → Graph optimization

Cloud-Based ML Processing (Offloaded Computation)

- FastAPI → Lightweight Python-based API for real-time communication
- TensorFlow Serving / ONNX Runtime Server → Deploy pre-trained ML models for:
  - Depth Estimation (MiDaS, DPT)
  - Object Detection (YOLOv8-Nano)
  - Scene Reconstruction (COLMAP)
- MQTT/WebSockets → For real-time data transfer between mobile and server
- PostgreSQL/PostGIS → Storing 3D maps and spatial data

On mobile (Real-time Rendering)

- PyOpenGL + Kivy → 3D rendering inside the mobile app
- pythreejs → If using Web-based 3D visualization
- Open3D → Point cloud visualization
- VisPy → Real-time graphics rendering

On Server (For More Complex 3D Mapping)

- Blender-Python → If detailed 3D environment models are needed
- COLMAP → 3D scene reconstruction from images
- PCL (Point Cloud Library) → Processing LiDAR/Depth data
