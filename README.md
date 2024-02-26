# Stereo Pose Estimation with ArUco Markers
This project aims to estimate the pose (rotation and translation) of a stereo camera setup using ArUco markers. ArUco markers are used as reference points to establish correspondences between the images captured by the left and right cameras, enabling the computation of the camera pose.

*Outline of epipolar lines gotten using the fundamental matrix*

![outline of epipolar geometry](https://github.com/IJAMUL1/RTDETR-Tracking-Detection/assets/60096099/cb7096f0-c06c-417f-9e2d-e2bea23e9be1)

## Objective
The primary objective of this project is to accurately determine the relative pose of the stereo camera system with respect to the scene. By detecting and analyzing ArUco markers in stereo images, the project seeks to provide robust and precise pose estimation, which can be valuable for applications such as robotics, augmented reality, and 3D reconstruction.

## Key Components
- ArUco Marker Detection: Utilizes the ArUco marker detection capabilities provided by OpenCV to identify markers in stereo images.

- Fundamental Matrix Estimation: Computes the fundamental matrix based on corresponding points detected in the left and right images. The fundamental matrix encapsulates the geometric relationship between the stereo image pair.

- Epipolar Line Calculation: Calculates epipolar lines for both images using the fundamental matrix. Epipolar lines represent the projection of points from one image onto the other image's plane.

- Pose Estimation: Estimates the camera's pose (rotation and translation) relative to the scene using the recovered essential matrix and camera intrinsic parameters.

- Visualization: Provides visualizations of the stereo images with overlaid epipolar lines to aid in understanding the pose estimation process.

## Usage
- Setup: Ensure all required dependencies, including OpenCV, are installed in your environment.

- Image Acquisition: Capture stereo images using a calibrated stereo camera setup.

- Run the Script: Execute the provided script stereo_pose_estimation.py after specifying the paths to the left and right stereo images.

- Visualization: View the output windows displaying the stereo images with overlaid epipolar lines and any printed pose estimation results.

## Results
The accuracy and reliability of the pose estimation results can be evaluated based on the alignment of epipolar lines and the consistency of the recovered camera pose with respect to ground truth or reference poses.

## Future Enhancements
Incorporate real-time pose estimation capabilities for dynamic scenes.

## Implement a graphical user interface (GUI) for user-friendly interaction and visualization.

Explore advanced techniques for robust marker detection and pose estimation under challenging conditions such as occlusions and varying lighting conditions.
