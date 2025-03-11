from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.snackbar import Snackbar

import cv2
import numpy as np
import os
import sys

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from slam_processor import SLAMProcessor
from ar_renderer import ARRenderer

class ARWidget(Image):
    def __init__(self, **kwargs):
        super(ARWidget, self).__init__(**kwargs)
        self.texture = None
        self.allow_stretch = True
        self.keep_ratio = False

    def update_frame(self, frame):
        """Update the widget with a new frame"""
        if frame is None:
            print("Warning: Received None frame in ARWidget.update_frame")
            return
        
        try:
            # Ensure frame is in the right format
            if frame.shape[2] == 4:  # RGBA
                colorfmt = 'rgba'
            elif frame.shape[2] == 3:  # RGB
                colorfmt = 'rgb'
            else:
                # Convert to RGB if not in RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                colorfmt = 'rgb'
                
            # Convert frame to texture
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt=colorfmt)
            texture.blit_buffer(buf, colorfmt=colorfmt, bufferfmt='ubyte')
            self.texture = texture
        except Exception as e:
            print(f"Error updating frame texture: {e}")
            # Create a simple error texture
            error_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(
                error_frame, 
                "Texture error", 
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 0, 0), 
                2
            )
            buf = cv2.flip(error_frame, 0).tobytes()
            texture = Texture.create(size=(320, 240), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.texture = texture

class StatusCard(MDCard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.size_hint = (None, None)
        self.size = (200, 120)
        self.pos_hint = {"top": 0.98, "right": 0.98}
        self.padding = 10
        self.spacing = 5
        self.elevation = 2
        self.md_bg_color = (0.9, 0.9, 0.9, 0.8)
        
        # Create labels for tracking info
        self.tracking_label = MDLabel(
            text="Tracking Quality: 100%",
            theme_text_color="Secondary",
            font_style="Body1"
        )
        self.features_label = MDLabel(
            text="Features: 0",
            theme_text_color="Secondary",
            font_style="Body1"
        )
        self.position_label = MDLabel(
            text="Position: [0, 0, 0]",
            theme_text_color="Secondary",
            font_style="Body1"
        )
        
        # Add labels to card
        self.add_widget(self.tracking_label)
        self.add_widget(self.features_label)
        self.add_widget(self.position_label)
        
    def update(self, tracking_quality, num_features, position):
        self.tracking_label.text = f"Tracking Quality: {tracking_quality*100:.1f}%"
        self.features_label.text = f"Features: {num_features}"
        self.position_label.text = f"Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]"

class ARNavigationApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_cls.primary_palette = "Blue"
        self.camera = None
        self.slam = SLAMProcessor()
        self.renderer = ARRenderer()
        
        # Camera parameters (adjust based on your camera)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ])
        
        # Debug mode flag
        self.debug_mode = False
        
    def build(self):
        # Create main layout
        layout = BoxLayout(orientation='vertical')
        
        # Create AR widget
        self.ar_widget = ARWidget()
        layout.add_widget(self.ar_widget)
        
        # Create button layout
        button_layout = BoxLayout(
            size_hint_y=None,
            height=50,
            spacing=10,
            padding=10
        )
        
        # Add reset button
        reset_button = MDRaisedButton(
            text="Reset SLAM",
            pos_hint={"center_x": 0.5},
            on_release=self.reset_slam
        )
        button_layout.add_widget(reset_button)
        
        # Add debug toggle button
        debug_button = MDRaisedButton(
            text="Toggle Debug",
            pos_hint={"center_x": 0.5},
            on_release=self.toggle_debug
        )
        button_layout.add_widget(debug_button)
        
        # Add button layout to main layout
        layout.add_widget(button_layout)
        
        # Create status card
        self.status_card = StatusCard()
        
        # Create root widget with FloatLayout for overlays
        from kivy.uix.floatlayout import FloatLayout
        root = FloatLayout()
        root.add_widget(layout)
        root.add_widget(self.status_card)
        
        # Initialize camera
        self.setup_camera()
        
        # Schedule frame updates
        Clock.schedule_interval(self.update_frame, 1.0/30.0)  # 30 FPS
        
        return root
        
    def setup_camera(self):
        """Initialize the camera capture"""
        try:
            # Try different camera backends and indices
            camera_backends = [
                (0, cv2.CAP_DSHOW),  # DirectShow on Windows
                (0, cv2.CAP_ANY),    # Default backend
                (1, cv2.CAP_ANY),    # Try second camera if available
                (0, cv2.CAP_MSMF)    # Media Foundation on Windows
            ]
            
            for idx, backend in camera_backends:
                print(f"Trying camera index {idx} with backend {backend}")
                if backend == cv2.CAP_ANY:
                    self.camera = cv2.VideoCapture(idx)
                else:
                    self.camera = cv2.VideoCapture(idx, backend)
                
                if self.camera.isOpened():
                    # Successfully opened camera
                    print(f"Successfully opened camera {idx} with backend {backend}")
                    break
            
            if not self.camera or not self.camera.isOpened():
                print("Error: Could not open any camera")
                return False
                
            # Set camera resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Read a test frame to ensure camera is working
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                print("Error: Camera opened but cannot read frames")
                return False
                
            print(f"Camera initialized with resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
            
            # Set camera parameters
            self.renderer.set_camera_parameters(self.camera_matrix, None)
            return True
        except Exception as e:
            print(f"Camera error: {str(e)}")
            return False
        
    def reset_slam(self, instance):
        """Reset SLAM tracking"""
        self.slam.reset()
        print("SLAM tracking reset")
        
    def toggle_debug(self, instance):
        """Toggle debug visualization mode"""
        self.debug_mode = not self.debug_mode
        status = "enabled" if self.debug_mode else "disabled"
        print(f"Debug mode {status}")
        
    def update_frame(self, dt):
        """Process and display each camera frame"""
        if not self.camera or not self.camera.isOpened():
            # If camera is not available, create a placeholder
            print("Camera not available or not opened")
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, 
                "Camera feed not available", 
                (640//2 - 150, 480//2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
            self.ar_widget.update_frame(placeholder)
            return
            
        # Read frame from camera
        ret, frame = self.camera.read()
        if not ret or frame is None:
            # If frame reading failed, create a placeholder
            print("Failed to read frame from camera")
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, 
                "Failed to read frame", 
                (640//2 - 150, 480//2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
            self.ar_widget.update_frame(placeholder)
            return
            
        # Convert frame to RGB
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"Frame shape: {frame_rgb.shape}")
            
            # Create a copy for display
            display_frame = frame_rgb.copy()
            
            # Process frame with SLAM (with error handling)
            try:
                keypoints, descriptors = self.slam.process_frame(frame_rgb, self.camera_matrix)
                
                # Get tracking quality metrics
                tracking_quality = self.slam.get_tracking_quality()
                num_features = len(keypoints) if keypoints is not None else 0
                position = self.slam.kf.x[0:3] if hasattr(self.slam, 'kf') else np.zeros(3)
                
                # Update status card
                self.status_card.update(tracking_quality, num_features, position)
                
                # Draw keypoints directly on the frame
                if keypoints is not None:
                    for kp in keypoints:
                        x, y = map(int, kp.pt)
                        cv2.circle(display_frame, (x, y), 3, (0, 255, 0), 1)
                
                # Draw tracking quality indicator
                quality_color = (
                    int(255 * (1.0 - tracking_quality)),  # Blue
                    int(255 * tracking_quality),          # Green
                    0                                     # Red
                )
                cv2.rectangle(
                    display_frame,
                    (10, display_frame.shape[0] - 30),
                    (10 + int(tracking_quality * 100), display_frame.shape[0] - 20),
                    quality_color,
                    -1
                )
            except Exception as slam_error:
                # If SLAM processing fails, just show the camera feed with an error message
                print(f"SLAM processing error: {slam_error}")
                cv2.putText(
                    display_frame, 
                    f"SLAM error: {str(slam_error)[:40]}", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
            
            # Update the widget with the processed frame
            self.ar_widget.update_frame(display_frame)
            
        except Exception as e:
            print(f"Error in update_frame: {e}")
            # If processing fails, create a placeholder with error message
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                error_frame, 
                f"Error: {str(e)[:40]}", 
                (10, 480//2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            self.ar_widget.update_frame(error_frame)
            
            # Use a direct message instead of Snackbar to avoid errors
            print(f"Frame processing error: {str(e)}")
            
    def on_stop(self):
        """Clean up resources when app closes"""
        if self.camera:
            self.camera.release()

if __name__ == '__main__':
    ARNavigationApp().run()