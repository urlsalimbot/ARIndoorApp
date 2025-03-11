import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
from kivy.graphics.texture import Texture

# Initialize GLUT
try:
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    GLUT_INITIALIZED = True
except Exception as e:
    print(f"Warning: Could not initialize GLUT: {e}")
    GLUT_INITIALIZED = False

class ARRenderer:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Navigation data
        self.current_position = np.zeros(3)
        self.current_orientation = np.eye(3)
        self.navigation_path = []
        self.waypoints = []
        self.path_confidence = []  # Confidence value for each path segment
        
        # AR objects and visual feedback
        self.ar_objects = []
        self.tracking_quality = 1.0  # Tracking quality indicator (0-1)
        self.show_debug = False  # Toggle debug visualization
        
        # Visual styles
        self.styles = {
            'path': {
                'color': (0.0, 1.0, 0.0, 0.7),  # Green
                'width': 2.0
            },
            'waypoint': {
                'color': (1.0, 0.0, 0.0, 0.7),  # Red
                'size': 0.1
            },
            'current_position': {
                'color': (0.0, 0.0, 1.0, 0.8),  # Blue
                'size': 0.15
            },
            'debug': {
                'color': (1.0, 1.0, 0.0, 0.5),  # Yellow
                'line_width': 1.0
            }
        }
        
        # Initialize textures for effects
        self.init_textures()
        
        # Store the current frame
        self.current_frame = None
        
    def init_textures(self):
        """Initialize textures for visual effects"""
        # Create arrow texture for direction indicators
        arrow_size = 32
        arrow_img = np.zeros((arrow_size, arrow_size, 4), dtype=np.uint8)
        cv2.arrowedLine(
            arrow_img,
            (arrow_size//4, arrow_size//2),
            (3*arrow_size//4, arrow_size//2),
            (255, 255, 255, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.3
        )
        
        self.arrow_texture = Texture.create(size=(arrow_size, arrow_size))
        self.arrow_texture.blit_buffer(arrow_img.tobytes(), colorfmt='rgba')
        
    def set_camera_parameters(self, camera_matrix, dist_coeffs):
        """Set camera intrinsic parameters"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
    def set_navigation_path(self, path, confidences=None):
        """Set the navigation path with optional confidence values"""
        self.navigation_path = path
        if confidences is None:
            self.path_confidence = [1.0] * (len(path) - 1)
        else:
            self.path_confidence = confidences
            
    def add_waypoint(self, position, label="", type="standard"):
        """Add a waypoint for navigation with type"""
        self.waypoints.append({
            'position': np.array(position),
            'label': label,
            'type': type,  # standard, destination, checkpoint
            'reached': False
        })
        
    def update_pose(self, position, orientation, tracking_quality=None):
        """Update current camera pose with tracking quality"""
        self.current_position = position
        self.current_orientation = orientation
        if tracking_quality is not None:
            self.tracking_quality = tracking_quality
            
    def project_point(self, point_3d):
        """Project 3D point to 2D image coordinates with distortion"""
        if self.camera_matrix is None:
            return None
            
        point_2d, _ = cv2.projectPoints(
            np.array([point_3d]), 
            cv2.Rodrigues(self.current_orientation)[0],
            self.current_position,
            self.camera_matrix,
            self.dist_coeffs
        )
        return point_2d[0][0]
        
    def setup_gl(self):
        """Setup OpenGL context with advanced settings"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Setup lighting for 3D objects
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        # Setup projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if self.camera_matrix is not None:
            fx = self.camera_matrix[0,0]
            fy = self.camera_matrix[1,1]
            cx = self.camera_matrix[0,2]
            cy = self.camera_matrix[1,2]
            near = 0.1
            far = 100.0
            
            # Compute proper field of view
            fovy = 2 * np.arctan2(cy, fy) * 180/np.pi
            aspect = float(self.width)/float(self.height)
            
            # Setup perspective projection
            glFrustum(
                near * (-cx) / fx,
                near * (self.width - cx) / fx,
                near * (-cy) / fy,
                near * (self.height - cy) / fy,
                near, far
            )
        else:
            gluPerspective(45, float(self.width)/float(self.height), 0.1, 100.0)
            
    def draw_path(self):
        """Draw navigation path with confidence-based styling"""
        if len(self.navigation_path) < 2:
            return
            
        glLineWidth(self.styles['path']['width'])
        
        # Draw path segments with confidence-based color
        glBegin(GL_LINE_STRIP)
        for i in range(len(self.navigation_path)-1):
            confidence = self.path_confidence[i]
            # Interpolate color based on confidence
            color = (
                self.styles['path']['color'][0],
                self.styles['path']['color'][1] * confidence,
                self.styles['path']['color'][2],
                self.styles['path']['color'][3] * confidence
            )
            glColor4f(*color)
            glVertex3f(*self.navigation_path[i])
            glVertex3f(*self.navigation_path[i+1])
        glEnd()
        
        # Draw direction arrows along the path
        if self.arrow_texture:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.arrow_texture.id)
            
            for i in range(0, len(self.navigation_path)-1, 3):
                start = np.array(self.navigation_path[i])
                end = np.array(self.navigation_path[i+1])
                direction = end - start
                length = np.linalg.norm(direction)
                
                if length > 0.1:
                    # Draw arrow billboard
                    self.draw_billboard_arrow(
                        (start + end) / 2,
                        np.arctan2(direction[1], direction[0])
                    )
                    
            glDisable(GL_TEXTURE_2D)
            
    def draw_billboard_arrow(self, position, angle):
        """Draw a billboard arrow that always faces the camera"""
        glPushMatrix()
        glTranslatef(*position)
        
        # Billboard rotation
        modelview = glGetFloatv(GL_MODELVIEW_MATRIX)
        camera_right = np.array([modelview[0][0], modelview[1][0], modelview[2][0]])
        camera_up = np.array([modelview[0][1], modelview[1][1], modelview[2][1]])
        
        # Rotate arrow
        glRotatef(angle * 180/np.pi, 0, 0, 1)
        
        glBegin(GL_QUADS)
        glColor4f(1, 1, 1, 0.8)
        size = 0.2
        glTexCoord2f(0, 0); glVertex3f(-size, -size, 0)
        glTexCoord2f(1, 0); glVertex3f(size, -size, 0)
        glTexCoord2f(1, 1); glVertex3f(size, size, 0)
        glTexCoord2f(0, 1); glVertex3f(-size, size, 0)
        glEnd()
        
        glPopMatrix()
        
    def draw_waypoints(self):
        """Draw waypoint markers with enhanced visualization"""
        for waypoint in self.waypoints:
            pos = waypoint['position']
            waypoint_type = waypoint['type']
            
            glPushMatrix()
            glTranslatef(*pos)
            
            # Different visualization based on waypoint type
            if waypoint_type == "destination":
                color = (1.0, 0.0, 0.0, 0.8)  # Red
                size = self.styles['waypoint']['size'] * 1.5
            elif waypoint_type == "checkpoint":
                color = (1.0, 0.5, 0.0, 0.8)  # Orange
                size = self.styles['waypoint']['size'] * 1.2
            else:
                color = self.styles['waypoint']['color']
                size = self.styles['waypoint']['size']
                
            # Draw waypoint marker
            glColor4f(*color)
            if waypoint['reached']:
                # Draw checkmark for reached waypoints
                self.draw_checkmark(size)
            else:
                glutSolidSphere(size, 12, 12)
            
            glPopMatrix()
            
            # Project and draw label with depth testing
            if waypoint['label']:
                point_2d = self.project_point(pos)
                if point_2d is not None:
                    glDisable(GL_DEPTH_TEST)
                    glRasterPos2f(point_2d[0], point_2d[1])
                    for c in waypoint['label']:
                        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(c))
                    glEnable(GL_DEPTH_TEST)
                    
    def draw_checkmark(self, size):
        """Draw a checkmark for reached waypoints"""
        glBegin(GL_LINES)
        glVertex3f(-size/2, 0, 0)
        glVertex3f(-size/6, -size/2, 0)
        glVertex3f(-size/6, -size/2, 0)
        glVertex3f(size/2, size/2, 0)
        glEnd()
        
    def draw_coordinate_frame(self):
        """Draw coordinate frame with enhanced visualization"""
        if not self.show_debug:
            return
            
        glLineWidth(self.styles['debug']['line_width'])
        glBegin(GL_LINES)
        # X axis - red
        glColor4f(1, 0, 0, 0.8)
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)
        # Y axis - green
        glColor4f(0, 1, 0, 0.8)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)
        # Z axis - blue
        glColor4f(0, 0, 1, 0.8)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)
        glEnd()
        
        # Draw axis labels
        if self.camera_matrix is not None:
            labels = [('X', (1.2, 0, 0)), ('Y', (0, 1.2, 0)), ('Z', (0, 0, 1.2))]
            for label, pos in labels:
                point_2d = self.project_point(pos)
                if point_2d is not None:
                    glRasterPos2f(point_2d[0], point_2d[1])
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(label))
                    
    def draw_tracking_quality_indicator(self):
        """Draw tracking quality indicator"""
        if not self.show_debug:
            return
            
        # Draw tracking quality bar
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Draw background bar
        glColor4f(0.2, 0.2, 0.2, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(110, 10)
        glVertex2f(110, 30)
        glVertex2f(10, 30)
        glEnd()
        
        # Draw quality level
        if self.tracking_quality > 0.7:
            color = (0.0, 1.0, 0.0, 0.8)  # Green
        elif self.tracking_quality > 0.4:
            color = (1.0, 1.0, 0.0, 0.8)  # Yellow
        else:
            color = (1.0, 0.0, 0.0, 0.8)  # Red
            
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(10 + 100 * self.tracking_quality, 10)
        glVertex2f(10 + 100 * self.tracking_quality, 30)
        glVertex2f(10, 30)
        glEnd()
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        
    def draw_frame(self):
        """Draw the current camera frame and return it"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            # If no frame is available, create a placeholder
            placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, 
                "Camera feed not available", 
                (self.width//2 - 150, self.height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
            return placeholder
        
        try:
            # Return a copy of the current frame with any overlays
            display_frame = self.current_frame.copy()
            
            # Draw keypoints if available
            if hasattr(self, 'current_keypoints') and self.current_keypoints is not None:
                for kp in self.current_keypoints:
                    x, y = map(int, kp.pt)
                    cv2.circle(display_frame, (x, y), 3, (0, 255, 0), 1)
            
            # Draw tracking quality indicator
            quality_color = (
                int(255 * (1.0 - self.tracking_quality)),  # Blue
                int(255 * self.tracking_quality),          # Green
                0                                          # Red
            )
            cv2.rectangle(
                display_frame,
                (10, self.height - 30),
                (10 + int(self.tracking_quality * 100), self.height - 20),
                quality_color,
                -1
            )
            
            return display_frame
        except Exception as e:
            print(f"Error drawing frame: {e}")
            # If processing fails, return a placeholder with error message
            error_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(
                error_frame, 
                f"Error: {str(e)[:40]}", 
                (10, self.height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            return error_frame

    def draw_debug_info(self, keypoints, tracking_quality):
        """Draw debug visualization
        
        Args:
            keypoints: List of detected keypoints
            tracking_quality: Current tracking quality (0-1)
        """
        if not self.show_debug or not hasattr(self, 'current_frame'):
            return
            
        # Draw keypoints
        glColor4f(*self.styles['debug']['color'])
        glLineWidth(self.styles['debug']['line_width'])
        for kp in keypoints:
            x, y = map(int, kp.pt)
            glBegin(GL_LINE_LOOP)
            for i in range(32):
                angle = 2 * np.pi * i / 32
                glVertex2f(x + 3 * np.cos(angle), y + 3 * np.sin(angle))
            glEnd()
                
        # Draw tracking quality indicator
        quality_color = (
            1.0 - tracking_quality,  # Red component
            tracking_quality,        # Green component
            0.0,                    # Blue component
            0.7                     # Alpha
        )
        glColor4f(*quality_color)
        glLineWidth(2)
        glBegin(GL_QUADS)
        glVertex2f(10, self.height - 30)
        glVertex2f(10 + tracking_quality * 100, self.height - 30)
        glVertex2f(10 + tracking_quality * 100, self.height - 20)
        glVertex2f(10, self.height - 20)
        glEnd()
        
    def update(self, frame, keypoints, pose=None):
        """Update the renderer with new frame and tracking data
        
        Args:
            frame: RGB frame from camera
            keypoints: List of detected keypoints
            pose: Current pose estimate from SLAM (if available)
        """
        # Store frame for rendering
        self.current_frame = frame
        self.current_keypoints = keypoints
        
        # Update pose if available
        if pose is not None:
            position = pose[0:3]
            # Simple orientation (identity for now)
            orientation = np.eye(3)
            self.update_pose(position, orientation)
            
    def render(self):
        """Main render function with enhanced visual feedback"""
        # Setup OpenGL state
        self.setup_gl()
        
        # Draw camera frame
        frame = self.draw_frame()
        
        # Draw navigation elements if we have tracking
        if hasattr(self, 'tracking_quality') and self.tracking_quality > 0.3:
            self.draw_path()
            self.draw_waypoints()
            self.draw_coordinate_frame()
            
        # Draw debug visualization if enabled
        if self.show_debug and hasattr(self, 'current_keypoints') and self.current_keypoints is not None:
            self.draw_debug_info(self.current_keypoints, self.tracking_quality)
