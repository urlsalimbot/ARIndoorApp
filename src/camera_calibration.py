import numpy as np
import cv2

class CameraCalibrator:
    def __init__(self, board_size=(9, 6), square_size=0.025):
        self.board_size = board_size
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
    def process_frame(self, frame):
        """Process a frame for calibration"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        
        if ret:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            
            # Store the points
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners2)
            
            # Draw the corners
            cv2.drawChessboardCorners(frame, self.board_size, corners2, ret)
            
            return True, frame
        return False, frame
        
    def calibrate(self, image_size):
        """Perform camera calibration"""
        if len(self.objpoints) < 10:
            return False, None, None
            
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None
        )
        
        if ret:
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            return True, mtx, dist
        return False, None, None
        
    def get_calibration_error(self):
        """Calculate reprojection error"""
        if self.camera_matrix is None:
            return None
            
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i],
                self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
            
        return mean_error/len(self.objpoints)
        
    def save_calibration(self, filename):
        """Save calibration parameters to file"""
        if self.camera_matrix is None:
            return False
            
        np.savez(
            filename,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            rvecs=self.rvecs,
            tvecs=self.tvecs
        )
        return True
        
    def load_calibration(self, filename):
        """Load calibration parameters from file"""
        try:
            data = np.load(filename)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.rvecs = data['rvecs']
            self.tvecs = data['tvecs']
            return True
        except:
            return False
