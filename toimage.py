import numpy as np
import open3d as o3d
import cv2
import os
import json
from matplotlib.path import Path
from scipy.spatial.transform import Rotation as R

# Normalize intensities
def normalize_intensities(intensities, cap=255):
    capped_intensities = np.minimum(intensities, cap)
    normalized_intensities = capped_intensities / cap
    return np.clip(normalized_intensities, 0, 1)

# Read PCD files with intensity values
def read_pcd_with_intensity(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file, remove_nan_points=True)
    points = np.asarray(pcd.points)
    if pcd.has_colors():
        intensities = np.asarray(pcd.colors)[:, 0] * 255  # Assuming intensity is stored in color
    else:
        intensities = np.ones((points.shape[0],)) * 255
    return points, intensities

# Fisheye projection of points
def project_points_fisheye(points, camera_matrix, dist_coeffs):
    if points.size == 0:
        return np.array([]), np.array([])

    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    image_points, _ = cv2.fisheye.projectPoints(points.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs)
    image_points = image_points.reshape(-1, 2)
    return image_points

# Transform points using the transformation matrix
def transform_points(points, T):
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed = (T @ points_hom.T).T[:, :3]
    return points_transformed

# Filter points that are inside a given polygon
def filter_points_in_polygon(image_points, polygon, corresponding_3d_points):
    path = Path(polygon)
    inside = path.contains_points(image_points)
    return image_points[inside], corresponding_3d_points[inside], inside

# Save colored point cloud as a PCD file
def save_colored_pcd(points, colors, output_pcd_file):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_pcd_file, pcd)
    print(f"Saved colored PCD at: {output_pcd_file}")

# Camera matrix and distortion coefficients
camera_matrix = np.array([
    [968.797410, 0.000000, 954.078232],
    [0.000000, 974.917054, 664.433449],
    [0.000000, 0.000000, 1.000000]
])
dist_coeffs = np.array([-0.037439, -0.004983, 0.004292, -0.002258])

# Load image and point cloud
base_folder = os.path.expanduser("~/Desktop/test4")
image_folder = os.path.join(base_folder, "images")
lidar_folder = os.path.join(base_folder, "lidar")
output_dir = os.path.join(base_folder, "output")
os.makedirs(output_dir, exist_ok=True)

image_file = os.path.join(image_folder, "camera_152port_a_cam_1-1688519870.610725218.png")
pcd_file = os.path.join(lidar_folder, "concatenated_point_cloud.pcd")

# Load the distorted image
image = cv2.imread(image_file)
if image is None:
    print(f"Error: Failed to load image from {image_file}")
    exit(1)

# Undistort the image
image_height, image_width = 1208, 1920
balance = 0
new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    camera_matrix, dist_coeffs, (image_width, image_height), np.eye(3), balance=balance
)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (image_width, image_height), cv2.CV_16SC2
)
rectified_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# Load and transform point cloud
points, intensities = read_pcd_with_intensity(pcd_file)
translation = np.array([0, -0.1, -0.3612])
yaw = -1.598
pitch = 0.0188
roll = -1.57
r = R.from_euler('ZYX', [yaw, pitch, roll])
rotation_matrix = r.as_matrix()
transformation_inv = np.eye(4)
transformation_inv[:3, :3] = rotation_matrix.T
transformation_inv[:3, 3] = -np.dot(rotation_matrix.T, translation)

points_transformed = transform_points(points, transformation_inv)

# Load polygon from JSON
json_file = os.path.join(base_folder, "camera_152port_a_cam_1-1688519870.610725218.json")
with open(json_file) as f:
    label_data = json.load(f)
polygon = np.array(label_data['shapes'][0]['points'])

# Project points and filter those inside the polygon
valid_indices = points_transformed[:, 2] > 0
points_camera_valid = points_transformed[valid_indices]
intensities_normalized = normalize_intensities(intensities[valid_indices])
image_points = project_points_fisheye(points_camera_valid, new_camera_matrix, np.zeros(4))

points_in_polygon_2d, points_in_polygon_3d, inside_mask = filter_points_in_polygon(image_points, polygon, points_camera_valid)

# Color points
colors = np.zeros((points_camera_valid.shape[0], 3))
colors[:, :] = intensities_normalized[:, None]  # Intensity-based color for all points
colors[inside_mask] = [1, 0, 0]  # Red color for points inside the polygon

# Save the colored point cloud
output_pcd_path = os.path.join(output_dir, "colored_lidar_152.pcd")
save_colored_pcd(points_camera_valid, colors, output_pcd_path)

# Overlay projected points onto the rectified image
for pt, inside in zip(image_points.astype(int), inside_mask):
    if 0 <= pt[0] < image_width and 0 <= pt[1] < image_height:
        color = (0, 0, 255) if inside else (0, 255, 0)
        cv2.circle(rectified_image, (pt[0], pt[1]), 2, color, -1)

# Save the final image
output_image_file = os.path.join(image_folder, "fisheye_rectified_projected.png")
cv2.imwrite(output_image_file, rectified_image)

print(f"Projected points onto rectified image saved to {output_image_file}")
