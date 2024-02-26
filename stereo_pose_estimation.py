import cv2
import numpy as np

def detect_aruco(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Load the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)
 
    # Draw markers on the image
    image_with_markers = cv2.aruco.drawDetectedMarkers(image.copy(), markerCorners, markerIds)

    return image_with_markers, markerCorners, markerIds

# Load two images
left = cv2.imread(r'pose_estimation_images\left.jpg')
right = cv2.imread(r'pose_estimation_images\right.jpg')

# Detect ArUco markers in the images
left_with_markers, corners1, ids1 = detect_aruco(left)
right_with_markers, corners2, ids2 = detect_aruco(right)

# Find corresponding points based on ArUco IDs using corner points directly
corresponding_points_left = []
corresponding_points_right = []

for id_left, corners_left in zip(ids1, corners1):
    for id_right, corners_right in zip(ids2, corners2):
        if id_left == id_right:
            # Use the corner points as the corresponding points
            corresponding_points_left.append(corners_left.reshape(-1, 2))  # Flatten to 2D array
            corresponding_points_right.append(corners_right.reshape(-1, 2))  # Flatten to 2D array

corresponding_points_left = np.concatenate(corresponding_points_left)
corresponding_points_right = np.concatenate(corresponding_points_right)

# Ensure both arrays are of type float32
corresponding_points_left = corresponding_points_left.astype(np.float32)
corresponding_points_right = corresponding_points_right.astype(np.float32)

# print(corresponding_points_left)

fundamental_matrix, _ = cv2.findFundamentalMat(corresponding_points_left, corresponding_points_right, cv2.FM_LMEDS)

# Print the fundamental matrix
print("Fundamental Matrix:")
print(fundamental_matrix)

# Draw epipolar lines on the left image
lines1 = cv2.computeCorrespondEpilines(corresponding_points_right.reshape(-1, 1, 2), 2, fundamental_matrix)
lines1 = lines1.reshape(-1, 3)

# Draw epipolar lines on the right image
lines2 = cv2.computeCorrespondEpilines(corresponding_points_left.reshape(-1, 1, 2), 1, fundamental_matrix)
lines2 = lines2.reshape(-1, 3)

# Draw epipolar lines on the left image
# img1_with_lines = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
# img1_with_lines = cv2.cvtColor(img1_with_lines, cv2.COLOR_GRAY2BGR)
img1_with_lines = left
for line in lines1:
    color = tuple(np.random.randint(0, 255, 3).tolist())
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [left.shape[1], -(line[2] + line[0] * left.shape[1]) / line[1]])
    cv2.line(img1_with_lines, (x0, y0), (x1, y1), color, 1)

# Draw epipolar lines on the right image
# img2_with_lines = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
# img2_with_lines = cv2.cvtColor(img2_with_lines, cv2.COLOR_GRAY2BGR)
img2_with_lines = right
for line in lines2:
    color = tuple(np.random.randint(0, 255, 3).tolist())
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [right.shape[1], -(line[2] + line[0] * right.shape[1]) / line[1]])
    cv2.line(img2_with_lines, (x0, y0), (x1, y1), color, 1)

# Scale the images for display
scale_factor = 0.4  # Adjust this value as needed
img1_with_lines_scaled = cv2.resize(img1_with_lines, None, fx=scale_factor, fy=scale_factor)
img2_with_lines_scaled = cv2.resize(img2_with_lines, None, fx=scale_factor, fy=scale_factor)


# Display the scaled images with epipolar lines
cv2.imshow('Left Image with Epipolar Lines', img1_with_lines_scaled)
cv2.imshow('Right Image with Epipolar Lines', img2_with_lines_scaled)

# Load the calibration results from the .npz file
calibration_data = np.load(r"calibration_result.npz")

# Access the saved variables (K matrix, distortion coefficients, etc.) by their names
K_matrix = calibration_data['mtx']
distortion_coefficients = calibration_data['dist'].reshape((-1,1))
# Compute the essential matrix
Ess_matrix = np.dot(np.dot(K_matrix.T, fundamental_matrix), K_matrix)

_, R, t, masks = cv2.recoverPose(Ess_matrix, corresponding_points_left, corresponding_points_right, K_matrix)

print("Rotation matrix =",R)
print("Translation matrix",t)


cv2.waitKey(0)
cv2.destroyAllWindows()
