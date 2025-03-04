import cv2
import numpy as np
import cv2.aruco as aruco
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *

class ARCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Get camera dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Camera parameters
        focal_length = max(self.width, self.height)
        center = [self.width/2, self.height/2]
        self.camera_matrix = np.array([[focal_length, 0, center[0]],
                                     [0, focal_length, center[1]],
                                     [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5,1))
        
        # Initialize Pygame for offscreen rendering
        pygame.init()
        self.surface = pygame.Surface((self.width, self.height))
        self.display = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | pygame.HIDDEN)
        
        # Set up OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up light
        glLight(GL_LIGHT0, GL_POSITION, (0, 0, 1, 0))
        glLight(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLight(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
        
        # Create display lists for cube
        self.cube_list = self.create_cube_list()
        
        # Create OpenCV window
        cv2.namedWindow('AR Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('AR Camera', self.width, self.height)
        
        # Store last known marker position and orientation
        self.last_rvec = None
        self.last_tvec = None
        self.marker_lost_time = 0
        self.interpolation_factor = 1.0
        self.marker_size = 0.05  # 5cm

    def create_cube_list(self):
        cube_list = glGenLists(1)
        glNewList(cube_list, GL_COMPILE)
        
        # Define vertices
        vertices = [
            [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
            [1, -1, 1], [1, 1, 1], [-1, -1, 1], [-1, 1, 1]
        ]
        
        # Define faces using vertices
        faces = [
            [0, 1, 2, 3],  # Back
            [4, 5, 6, 7],  # Front
            [1, 5, 7, 2],  # Top
            [0, 4, 6, 3],  # Bottom
            [0, 1, 5, 4],  # Right
            [3, 2, 7, 6]   # Left
        ]
        
        # Define colors for each face
        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0]   # Cyan
        ]
        
        glBegin(GL_QUADS)
        for face_id, face in enumerate(faces):
            glColor3fv(colors[face_id])
            for vertex_id in face:
                glVertex3fv(vertices[vertex_id])
        glEnd()
        
        glEndList()
        return cube_list

    def smooth_transition(self, current_time):
        if self.marker_lost_time > 0:
            time_since_lost = current_time - self.marker_lost_time
            self.interpolation_factor = max(0, 1 - time_since_lost)
            
            if self.interpolation_factor <= 0:
                self.last_rvec = np.array([[0], [0], [0]], dtype=np.float32)
                self.last_tvec = np.array([[0], [0], [0.5]], dtype=np.float32)

    def render_opengl(self, frame, rvec=None, tvec=None):
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.width/self.height, 0.1, 100.0)
        
        # Use last known position if no new position is provided
        if rvec is None and tvec is None:
            if self.last_rvec is not None:
                rvec = self.last_rvec
                tvec = self.last_tvec
            else:
                rvec = np.array([[0], [0], [0]], dtype=np.float32)
                tvec = np.array([[0], [0], [0.5]], dtype=np.float32)
        else:
            self.last_rvec = rvec
            self.last_tvec = tvec
            self.marker_lost_time = 0
            self.interpolation_factor = 1.0
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Create OpenGL modelview matrix that aligns cube with marker
        align_rotation = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        R = R @ align_rotation
        
        modelview = np.array([
            [R[0,0], R[0,1], R[0,2], tvec[0,0]],
            [R[1,0], R[1,1], R[1,2], tvec[1,0]],
            [R[2,0], R[2,1], R[2,2], tvec[2,0]],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Set modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLoadMatrixf(modelview.T)
        
        # Move cube above marker and scale it
        glTranslatef(0, 0, self.marker_size * 0.5)
        glScalef(self.marker_size, self.marker_size, self.marker_size)
        
        # Apply smooth transition when marker is lost
        if self.marker_lost_time > 0:
            glRotatef(90 * (1.0 - self.interpolation_factor), 0, 1, 0)
        
        # Draw cube
        glCallList(self.cube_list)
        
        # Read the OpenGL buffer
        glPixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        gl_surface = np.frombuffer(glPixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        gl_surface = cv2.flip(gl_surface, 0)
        
        # Create alpha mask from the OpenGL rendering
        gray = cv2.cvtColor(gl_surface, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Composite OpenGL rendering onto camera frame
        gl_surface = cv2.cvtColor(gl_surface, cv2.COLOR_RGB2BGR)
        np.copyto(frame, gl_surface, where=mask.astype(bool))

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Convert to grayscale for marker detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        current_time = pygame.time.get_ticks() / 1000.0
        
        if ids is not None and len(corners[0][0]) == 4:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Calculate marker center
            marker_corners = corners[0][0]
            marker_center = np.mean(marker_corners, axis=0)
            
            # Use the first marker as reference
            objPoints = np.array([[-self.marker_size/2, -self.marker_size/2, 0],
                                [self.marker_size/2, -self.marker_size/2, 0],
                                [self.marker_size/2, self.marker_size/2, 0],
                                [-self.marker_size/2, self.marker_size/2, 0]], dtype=np.float32)
            
            success, rvec, tvec = cv2.solvePnP(objPoints, 
                                              corners[0][0], 
                                              self.camera_matrix, 
                                              self.dist_coeffs,
                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
            
            if success:
                self.render_opengl(frame, rvec, tvec)
        else:
            if self.marker_lost_time == 0:
                self.marker_lost_time = current_time
            
            self.smooth_transition(current_time)
            self.render_opengl(frame)
        
        return frame

    def run(self):
        while True:
            frame = self.process_frame()
            if frame is None:
                break
            
            cv2.imshow('AR Camera', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    ar_camera = ARCamera()
    ar_camera.run()