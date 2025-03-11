import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial import cKDTree
import time

class SLAMProcessor:
    def __init__(self):
        # ORB detector parameters
        self.orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # Initialize Kalman filter for 3D position and velocity
        self.kf = KalmanFilter(dim_x=6, dim_z=3)  # State: [x, y, z, vx, vy, vz]
        dt = 1.0/30.0  # Assuming 30fps
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R *= 0.1
        
        # Process noise
        self.kf.Q *= 0.1
        
        # Initial state
        self.kf.x = np.zeros(6)
        
        # Feature tracking
        self.prev_kp = None
        self.prev_desc = None
        self.prev_time = None
        
        # Keyframe management
        self.keyframes = []  # List of (keypoints, descriptors, pose) tuples
        self.keyframe_tree = None
        self.min_keyframe_distance = 0.5  # Minimum distance between keyframes
        
        # Tracking quality metrics
        self.min_features = 50
        self.good_features = 500
        self.min_matches = 10
        self.good_matches = 50
        self.tracking_quality = 1.0
        self.feature_ratio = 1.0
        self.match_ratio = 1.0
        self.motion_smoothness = 1.0
        self.last_positions = []
        self.max_positions = 10
        
    def reset(self):
        """Reset SLAM processor state"""
        self.kf.x = np.zeros(6)
        self.prev_kp = None
        self.prev_desc = None
        self.prev_time = None
        self.keyframes = []
        self.keyframe_tree = None
        self.tracking_quality = 1.0
        self.feature_ratio = 1.0
        self.match_ratio = 1.0
        self.motion_smoothness = 1.0
        self.last_positions = []
        
    def get_tracking_quality(self):
        """Get overall tracking quality metric (0-1)"""
        # Combine different quality metrics with weights
        weights = {
            'feature_ratio': 0.3,
            'match_ratio': 0.3,
            'motion_smoothness': 0.4
        }
        
        quality = (
            weights['feature_ratio'] * self.feature_ratio +
            weights['match_ratio'] * self.match_ratio +
            weights['motion_smoothness'] * self.motion_smoothness
        )
        
        # Smooth the quality metric
        self.tracking_quality = 0.8 * self.tracking_quality + 0.2 * quality
        return self.tracking_quality
        
    def update_tracking_metrics(self, num_features, num_matches, position):
        """Update tracking quality metrics"""
        # Update feature ratio
        self.feature_ratio = min(1.0, num_features / self.good_features)
        
        # Update match ratio
        if self.prev_kp is not None:
            self.match_ratio = min(1.0, num_matches / self.good_matches)
        
        # Update motion smoothness based on position history
        self.last_positions.append(position)
        if len(self.last_positions) > self.max_positions:
            self.last_positions.pop(0)
            
        if len(self.last_positions) >= 3:
            velocities = np.diff(self.last_positions, axis=0)
            accelerations = np.diff(velocities, axis=0)
            smoothness = np.exp(-np.mean(np.linalg.norm(accelerations, axis=1)))
            self.motion_smoothness = 0.7 * self.motion_smoothness + 0.3 * smoothness
        
    def detect_features(self, frame):
        """Detect ORB features in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)
        return kp, desc
        
    def match_features(self, desc1, desc2, ratio=0.75):
        """Match features using kNN matching with ratio test"""
        if desc1 is None or desc2 is None:
            return []
            
        # Create BFMatcher and match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            matches = bf.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match in matches:
                if len(match) == 2:  # Check if we have two matches
                    m, n = match
                    if m.distance < ratio * n.distance:
                        good_matches.append(m)
                elif len(match) == 1:  # If we only have one match
                    good_matches.append(match[0])
                    
            return good_matches
        except Exception as e:
            print(f"Error in feature matching: {e}")
            # Fallback to simple matching if knnMatch fails
            try:
                matches = bf.match(desc1, desc2)
                # Sort matches by distance
                matches = sorted(matches, key=lambda x: x.distance)
                # Take top 50 matches
                return matches[:50] if len(matches) > 50 else matches
            except Exception as e2:
                print(f"Fallback matching also failed: {e2}")
                return []
        
    def estimate_pose(self, kp1, kp2, matches, K):
        """Estimate relative pose between frames"""
        if len(matches) < self.min_matches:
            return None, None
            
        # Get matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None
            
        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        
        return R, t
        
    def process_frame(self, frame, K):
        """Process a new frame"""
        # Detect features
        kp, desc = self.detect_features(frame)
        
        # Initialize if first frame
        if self.prev_kp is None:
            self.prev_kp = kp
            self.prev_desc = desc
            self.prev_time = time.time()
            return kp, desc
            
        # Match features with previous frame
        matches = self.match_features(self.prev_desc, desc)
        
        # Estimate pose if enough matches
        if len(matches) >= self.min_matches:
            R, t = self.estimate_pose(self.prev_kp, kp, matches, K)
            
            if R is not None and t is not None:
                # Update Kalman filter
                dt = time.time() - self.prev_time
                self.kf.F[0:3, 3:6] = np.eye(3) * dt
                
                # Predict
                self.kf.predict()
                
                # Update with measured position
                scale = 0.1  # Scale factor for translation
                measured_pos = self.kf.x[0:3] + scale * t.flatten()
                self.kf.update(measured_pos)
                
                # Update tracking metrics
                self.update_tracking_metrics(len(kp), len(matches), self.kf.x[0:3])
                
                # Add keyframe if moved enough
                if len(self.keyframes) == 0:
                    self.keyframes.append((kp, desc, self.kf.x[0:3]))
                else:
                    dist = np.linalg.norm(self.kf.x[0:3] - self.keyframes[-1][2])
                    if dist > self.min_keyframe_distance:
                        self.keyframes.append((kp, desc, self.kf.x[0:3]))
                        
                        # Update keyframe tree
                        positions = np.array([kf[2] for kf in self.keyframes])
                        self.keyframe_tree = cKDTree(positions)
        else:
            # Poor tracking, increase uncertainty
            self.kf.P *= 1.5
            self.tracking_quality *= 0.8
            
        # Update previous frame info
        self.prev_kp = kp
        self.prev_desc = desc
        self.prev_time = time.time()
        
        return kp, desc
