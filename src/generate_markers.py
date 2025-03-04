import cv2
import numpy as np

def generate_aruco_markers():
    # Initialize the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Generate 4 markers
    for marker_id in range(4):
        # Create the marker
        marker_size = 400  # pixels
        marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_image, 1)
        
        # Save the marker
        filename = f'marker_{marker_id}.png'
        cv2.imwrite(filename, marker_image)
        print(f'Generated {filename}')

if __name__ == "__main__":
    generate_aruco_markers()
