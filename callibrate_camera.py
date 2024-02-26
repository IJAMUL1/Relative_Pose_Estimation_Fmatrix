import numpy as np
import cv2
import glob

# Termination criteria for subpixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 8,3), np.float32)
objp[:, :2] = np.mgrid[0:8,0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D point in real-world space
imgpoints = []  # 2D points in the image plane.

images = glob.glob(r'callibration_images\*.jpg')  # Replace with your image directory

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    # If found, add object points and image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Refine corner positions for better accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
if ret:
    print("Calibration successful")
else:
    print("Calibration failed")
    
print(mtx)

print(dist)


# Save calibration results
np.savez(r'calibration_result.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

