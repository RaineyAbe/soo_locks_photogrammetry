#!/usr/bin/env python
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


def calibrate_camera(
        image_file: str = None, 
        gcp_df = None, 
        dem_file: str = None, 
        out_folder: str = None,
        plot_results: bool = True
        ):
    # --- Load inputs ---
    # Image
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]
    dim = w,h
    # DEM
    dem = rxr.open_rasterio(dem_file).squeeze()
    dem = xr.where(dem == -9999, np.nan, dem)

    # --- Calibrate camera intrinsics (distortion parameters) ---
    print('Calibrating camera intrinsics (distortion parameters)')

    # Compile GCP (object) and image (pixel) points
    objp = gcp_df[['X', 'Y', 'Z']].values.astype(np.float32).reshape(-1, 1, 3)
    imgp = gcp_df[['col_sample', 'row_sample']].values.astype(np.float32).reshape(-1, 1, 2)

    if len(objp) < 4:
        raise ValueError("Need at least 4 GCPs per image for fisheye calibration")

    # Center object points for numerical stability
    objp_mean = objp.mean(axis=0)
    objp_centered = objp - objp_mean
    objpoints = [objp_centered]
    imgpoints = [imgp]

    # Initialize intrinsics
    fx = fy = 2000
    cx = w / 2
    cy = h / 2
    K_init = np.array([
        [fx,0,cx],
        [0,fy,cy],
        [0,0,1]
        ], dtype=np.float64)
    D_init = np.zeros((4, 1))  # fisheye has 4 distortion coefficients

    # Calibration flags
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_CHECK_COND +
        cv2.fisheye.CALIB_FIX_SKEW
        )

    # Run calibration
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, dim, K_init, D_init, None, None, flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    )

    # Undistort image
    K_full = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=1)
    # catch wacky solutions
    if K_full[0][0] < 1:
        K_full = K.copy()
        K_full[0,0] = K[0,0]/2
        K_full[1,1] = K[1,1]/2
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_full, dim, cv2.CV_32FC1)
    undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Mask invalid pixels
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask_undistorted = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST)
    valid_mask = mask_undistorted > 0
    undistorted = undistorted.astype(np.float32)
    undistorted[~valid_mask] = np.nan

    # Undistort GCP points
    undistorted_pts = cv2.fisheye.undistortPoints(imgp, K, D, P=K_full).reshape(-1, 2)
    gcp_df['col_sample_undistorted'] = undistorted_pts[:, 0]
    gcp_df['row_sample_undistorted'] = undistorted_pts[:, 1]

    # Calculate rectified extrinsics
    rvec = rvecs[0]
    tvec = tvecs[0] - cv2.Rodrigues(rvec)[0] @ objp_mean.reshape(3,1) # un-center translation

    return K, D, K_full, rvec, tvec, gcp_df


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
    ortho_xr = ortho_xr.dropna(dim='x', how='all').dropna(dim='y', how='all')

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


class GCPSelector:
    def __init__(self, img1_file, img2_file, gcp_uv, zoom=100):
        """
        img1_file, img2_file: paths to grayscale images
        gcp_uv: Nx2 array of (col_sample, row_sample)
        zoom: half-size (pixels) of window to show around each GCP
        """
        # Load grayscale images
        self.img1 = plt.imread(img1_file)
        if self.img1.ndim == 3:
            self.img1 = np.mean(self.img1, axis=2)
        self.img2 = plt.imread(img2_file)
        if self.img2.ndim == 3:
            self.img2 = np.mean(self.img2, axis=2)

        self.gcp_uv = np.array(gcp_uv)
        self.zoom = zoom
        self.clicked_points = [None] * len(self.gcp_uv)
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
        if self.current_index >= len(self.gcp_uv):
            print("All GCPs processed.")
            plt.close(self.fig)
            return

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()

        u, v = self.gcp_uv[self.current_index]
        u, v = int(u), int(v)
        half = self.zoom

        # Define crop bounds
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
        # print(f"Selected GCP #{self.current_index} new location: ({u:.1f}, {v:.1f})")
        self.clicked_points[self.current_index] = (u, v)
        self.current_index += 1
        self.update_display()

    def onkey(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'k':  # skip
            # print(f"Skipped GCP #{self.current_index}")
            self.clicked_points[self.current_index] = None
            self.current_index += 1
            self.update_display()
        elif event.key == 'b':  # back
            if self.current_index > 0:
                self.current_index -= 1
                # print(f"Going back to GCP #{self.current_index}")
                self.update_display()
        elif event.key == 'q':
            print("User quit early.")
            plt.close(self.fig)

    def get_results(self):
        """Return (original_pts, clicked_pts), with skipped GCPs as NaN"""
        valid_idx = np.arange(len(self.gcp_uv))
        clicked = np.array([
            [np.nan, np.nan] if c is None else c for c in self.clicked_points
        ])
        return self.gcp_uv, clicked


# SETUP
data_folder = '/Users/rdcrlrka/Research/Soo_locks'
og_image_list = sorted(glob.glob(os.path.join(data_folder, 'camera_calibration', 'single_band_images', '*.tiff')))
new_image_list = sorted(glob.glob(os.path.join(data_folder, 'camera_calibration', 'images20251021', '*.tiff')))
print(f"Found {len(og_image_list)} old and {len(new_image_list)} new images.")

# Load calibration (not used directly, but still helpful context)
calib_file = os.path.join(data_folder, 'camera_calibration', 'calibration_params', 'calibration_parameters_merged.csv')
calib = pd.read_csv(calib_file)
for k in ['K', 'D', 'K_full', 'R_rectified', 't_rectified']:
    calib[k] = calib[k].apply(literal_eval)

# Load GCPs
gcp_file = os.path.join(data_folder, 'inputs', 'gcp', 'GCP_merged_stable.gpkg')
gcp = gpd.read_file(gcp_file, layer='gcp_merged')
gcp = gcp.dropna().reset_index(drop=True)
gcp['channel'] = [f"ch0{ch}" if ch < 10 else f"ch{ch}" for ch in gcp['channel']]

# TEST WITH ONE CAMERA
cam_index = 0
params = calib.iloc[cam_index]
img1_file = og_image_list[cam_index]
img2_file = new_image_list[cam_index]

# Match GCPs to this camera
ch = f"ch{os.path.basename(img1_file).split('ch')[1][0:2]}"
gcp_img1 = gcp.loc[gcp['channel'] == ch]
gcp_uv = gcp_img1[['col_sample', 'row_sample']].values

# Run interactive GCP selection with zoomed view
selector = GCPSelector(
    img1_file=img1_file,
    img2_file=img2_file,
    gcp_uv=gcp_uv,
    zoom=100  # increase/decrease as needed
)

orig_pts, new_pts = selector.get_results()

# Save results
out_file = os.path.join(data_folder, f"camera_{cam_index:02d}_gcp_matches.csv")
df = pd.DataFrame({
    'orig_u': orig_pts[:, 0],
    'orig_v': orig_pts[:, 1],
    'new_u': new_pts[:, 0] if len(new_pts) > 0 else np.nan,
    'new_v': new_pts[:, 1] if len(new_pts) > 0 else np.nan,
})
print(df)
# df.to_csv(out_file, index=False)
# print(f"Saved {len(new_pts)} GCP matches to {out_file}")
