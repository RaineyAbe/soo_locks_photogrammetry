#! /usr/bin/env python

import os
import glob
import pandas as pd
import geopandas as gpd
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np
import cv2
import rioxarray as rxr
import xarray as xr
from scipy.spatial.transform import Rotation as R


class GCPSelector:
    def __init__(self, img1_file, img2_file, gcp1, zoom=100):
        # Load grayscale images
        self.img1 = plt.imread(img1_file)
        if self.img1.ndim == 3:
            self.img1 = np.mean(self.img1, axis=2)
        self.img2 = plt.imread(img2_file)
        if self.img2.ndim == 3:
            self.img2 = np.mean(self.img2, axis=2)

        self.gcp1 = gcp1
        self.gcp2 = gcp1.copy()
        self.zoom = zoom
        self.clicked_points = [None] * len(self.gcp2)
        self.current_index = 0

        # Setup figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkey)

        print(
            "Instructions:\n"
            " - Click on the right image to record a new GCP location.\n"
            " - Press 'k' to skip if feature not visible.\n"
            " - Press 'b' to go back and redo previous GCP.\n"
            " - Close window when done."
        )

        self.update_display()
        plt.show()

    def update_display(self):
        """Show zoomed-in patches around current GCP"""
        if self.current_index >= len(self.gcp1):
            print("All GCPs processed.")
            plt.close(self.fig)
            return

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()

        u, v = self.gcp1.iloc[self.current_index][['col_sample','row_sample']]
        u, v = int(u), int(v)
        half = self.zoom

        # Crop images to around GCP
        x1, x2 = max(0, u - half), min(self.img1.shape[1], u + half)
        y1, y2 = max(0, v - half), min(self.img1.shape[0], v + half)
        crop1 = self.img1[y1:y2, x1:x2]
        crop2 = self.img2[y1:y2, x1:x2]

        # Show zoomed-in crops
        self.ax1.imshow(crop1, cmap='gray', extent=[x1, x2, y2, y1])
        self.ax1.set_title(f"Original GCP #{self.current_index}")
        self.ax1.plot(u, v, 'om', markersize=6)

        self.ax2.imshow(crop2, cmap='gray', extent=[x1, x2, y2, y1])
        self.ax2.set_title("Click New Location")

        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(u - half, u + half)
            ax.set_ylim(v + half, v - half)

        self.fig.canvas.draw()

    def onclick(self, event):
        if event.inaxes != self.ax2:
            return
        u, v = event.xdata, event.ydata
        self.clicked_points[self.current_index] = (u, v)
        self.gcp2.loc[self.current_index, ['col_sample', 'row_sample']] = u, v
        self.current_index += 1
        self.update_display()

    def onkey(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'k':  # skip
            self.clicked_points[self.current_index] = None
            self.current_index += 1
            self.update_display()
        elif event.key == 'b':  # back
            if self.current_index > 0:
                self.current_index -= 1
                self.update_display()
        elif event.key == 'q':
            print("User quit early.")
            plt.close(self.fig)

    def get_results(self):
        """Return GCP2 with updated pixel coordinates, with skipped GCPs as NaN"""
        gcp1 = self.gcp1
        gcp2 = self.gcp2
        # remove NaN rows
        ikeep = gcp2.notna().any(axis=1)
        gcp1 = gcp1[ikeep].reset_index(drop=True)
        gcp2 = gcp2[ikeep].reset_index(drop=True)

        return gcp1, gcp2


def refine_camera_pose(gcp_df, K, D, rvec1, tvec1):
    # Prepare GCPs and image points
    obj_pts = gcp_df[['X', 'Y', 'Z']].values.astype(np.float32)
    img_pts = gcp_df[['col_sample', 'row_sample']].values.astype(np.float32)

    # Undistort the image points
    img_pts_undistorted = cv2.fisheye.undistortPoints(
        img_pts.reshape(-1, 1, 2), K, D, None, K
    ).reshape(-1, 2)

    # Convert rvec1, tvec1 to rotation matrix
    R1, _ = cv2.Rodrigues(rvec1)
    tvec1 = tvec1.reshape(3, 1)

    # Project GCPs into the camera frame using initial pose
    # world -> camera
    obj_cam = (R1 @ obj_pts.T + tvec1).T

    # Triangulate new camera-frame coordinates based on observed pixels
    # Backproject undistorted pixels into 3D rays, then scale by depth of old model
    pts_cam_dir = np.concatenate([img_pts_undistorted, np.ones((len(img_pts_undistorted), 1))], axis=1)
    pts_cam_dir = np.linalg.inv(K) @ pts_cam_dir.T
    pts_cam_dir = pts_cam_dir.T

    # Use existing depths from old pose to approximate new camera-frame coordinates
    depths = obj_cam[:, 2:3]
    obj_cam_new = pts_cam_dir * depths

    # Compute best-fit rigid transform (dR, dt) between obj_cam_new and obj_cam
    mu_old = obj_cam.mean(axis=0)
    mu_new = obj_cam_new.mean(axis=0)
    X0 = obj_cam - mu_old
    X1 = obj_cam_new - mu_new

    U, _, Vt = np.linalg.svd(X0.T @ X1)
    R_delta = Vt.T @ U.T
    if np.linalg.det(R_delta) < 0:  # ensure a right-handed rotation
        Vt[-1, :] *= -1
        R_delta = Vt.T @ U.T
    t_delta = mu_new - R_delta @ mu_old

    # Construct new pose
    R2 = R_delta @ R1
    tvec2 = R_delta @ tvec1 + t_delta.reshape(3, 1)
    rvec2, _ = cv2.Rodrigues(R2)

    return rvec2, tvec2


def undistort_image(img, K, D, K_full):
    h,w = img.shape
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_full, (w,h), cv2.CV_32FC1)
    img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Mask invalid pixels
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask_undistorted = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST)
    valid_mask = mask_undistorted > 0
    img_undistorted = img_undistorted.astype(np.float32)
    img_undistorted[~valid_mask] = np.nan

    return img_undistorted


def orthorectify(image_file, dem_file, K, D, K_full, rvec, tvec, out_file=None):
    # Undistort the image
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img_undistorted = undistort_image(image, K, D, K_full)

    # Load DEM
    dem = rxr.open_rasterio(dem_file).squeeze()
    crs = dem.rio.crs
    dem = xr.where(dem==-9999, np.nan, dem)
   
    # Build world coordinates for valid DEM pixels
    dem_z = dem.data
    X, Y = np.meshgrid(dem.x.data, dem.y.data)
    valid_dem_mask = np.isfinite(dem_z)
    world_pts = np.stack([X[valid_dem_mask], Y[valid_dem_mask], dem_z[valid_dem_mask]], axis=-1)

    # Project valid DEM points
    img_pts, _ = cv2.projectPoints(world_pts, cv2.Rodrigues(rvec)[0], tvec, K_full, None)
    map_x = np.full_like(dem_z, np.nan, dtype=np.float32)
    map_y = np.full_like(dem_z, np.nan, dtype=np.float32)
    map_x[valid_dem_mask] = img_pts[:, 0, 0]
    map_y[valid_dem_mask] = img_pts[:, 0, 1]

    # Remap undistorted image to DEM grid
    ortho = np.full_like(dem_z, np.nan, dtype=np.float32)
    in_bounds = (
        (map_x >= 0) & (map_x < img_undistorted.shape[1]) &
        (map_y >= 0) & (map_y < img_undistorted.shape[0])
    )
    sampled = cv2.remap(
        img_undistorted, 
        map_x.astype(np.float32), 
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR
        )
    ortho[in_bounds] = sampled[in_bounds]
    ortho = np.where(np.isnan(dem_z), np.nan, ortho)

    # Convert to DataArray
    ortho_xr = xr.DataArray(
        data=ortho,
        dims=('y', 'x'),
        coords={'x': dem.x.data, 'y': dem.y.data}
    )
    # remove all-NaN rows and columns
    # ortho_xr = ortho_xr.dropna(dim='x', how='all').dropna(dim='y', how='all')

    # Save to file
    if out_file:
        # convert datatype to int
        nodata_val = 9999
        ortho_xr = xr.where(np.isnan(ortho_xr), nodata_val, ortho_xr)
        ortho_xr = ortho_xr.astype(np.uint16)
        # assign CRS and nodata
        ortho_xr = ortho_xr.rio.write_nodata(nodata_val)
        ortho_xr = ortho_xr.rio.write_crs(crs)
        # ensure orientation matches DEM (north-up)
        if dem.rio.resolution()[1] < 0:
            ortho_xr = ortho_xr.sortby('y', ascending=False)
        # save with integer compression
        ortho_xr.rio.to_raster(
            out_file,
            compress='lzw',
            dtype='uint16'
        )
        print(f"Orthorectified image saved to:\n{out_file}")

    return ortho_xr


# SETUP
data_folder = '/Users/rdcrlrka/Research/Soo_locks'
og_image_list = sorted(glob.glob(os.path.join(data_folder, 'camera_calibration', 'IR_concurrent_with_lidar', '*.tiff')))
new_image_list = sorted(glob.glob(os.path.join(data_folder, 'camera_calibration', 'images20251021', '*.tiff')))
print(f"Found {len(og_image_list)} original and {len(new_image_list)} new images.")
refdem_file = os.path.join(data_folder, 'inputs', 'lidar_DSM_filled_cropped.tif')
out_folder = os.path.join(data_folder, 'camera_recalibration_testing')
os.makedirs(out_folder, exist_ok=True)

# Load initial calibration parameters
calib_file = os.path.join(data_folder, 'inputs', 'original_calibrated_cameras.csv')
calib = pd.read_csv(calib_file)
for k in ['K', 'D', 'K_full', 'rvec', 'tvec']:
    calib[k] = calib[k].apply(literal_eval)

# Load GCPs
gcp_file = os.path.join(data_folder, 'inputs', 'GCP_merged_stable.gpkg')
gcp = gpd.read_file(gcp_file, layer='gcp_merged')
gcp = gcp.dropna().reset_index(drop=True)
gcp['channel'] = [f"ch0{ch}" if ch < 10 else f"ch{ch}" for ch in gcp['channel']]

# Iterate over images
for i,img1_file in enumerate(og_image_list):
    img2_file = new_image_list[i]

    # Subset GCPs to this camera
    ch = f"ch{os.path.basename(img1_file).split('ch')[1][0:2]}"
    gcp1 = gcp.loc[gcp['channel'] == ch].reset_index(drop=True)

    # Get camera intrinsics
    cam = calib.loc[calib['channel']==ch]
    K = np.array(cam['K'].values[0])
    D = np.array(cam['D'].values[0])
    K_full = np.array(cam['K_full'].values[0])
    rvec1 = np.array(cam['rvec'].values[0])
    tvec1 = np.array(cam['tvec'].values[0])

    # Run interactive GCP selection
    gcp2_file = os.path.join(out_folder, f"{ch}_updated_GCP.gpkg")
    if not os.path.exists(gcp2_file):

        selector = GCPSelector(
            img1_file, img2_file, gcp1, zoom=100
        )

        # Get the updated GCP
        gcp1, gcp2 = selector.get_results()
        if len(gcp2) < 4:
            raise ValueError("Not enough GCPs for calibration")
    else:
        gcp2 = gpd.read_file(gcp2_file)

    # Solve new camera parameters
    rvec2, tvec2 = refine_camera_pose(
        image_file = img2_file, gcp_df=gcp2, K=K, D=D, rvec1=rvec1, tvec1=tvec1
    )
    #calibrate_camera(img2_file, gcp2, K, D, rvec1, tvec1)
    
    # Orthorectify image 2
    out_file = os.path.join(out_folder, f"{ch}_orthoimage.tiff")
    orthorectify(
        img2_file, refdem_file, K, D, K_full, rvec2, tvec2, out_file
    )