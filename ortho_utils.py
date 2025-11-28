#! /usr/bin/env python

"""
Utility functions for generating an orthoimage from video files for the Soo Locks project.
"""

import os
from glob import glob
import numpy as np
import cv2
import rioxarray as rxr
import xarray as xr
import rasterio as rio
import datetime
import dask.array as da
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import json
from tqdm import tqdm
import functools
# Ignore warnings (rasterio throws a warning whenever an image is not georeferenced. Annoying in this case.)
import warnings
warnings.filterwarnings('ignore')


def string_to_datetime(
        datetime_string: str = None
        ):
    """
    Convert a datetime string in the format 'YYYYMMDD_HHMMSS' to a datetime object.
    
    Parameters
    ----------
    datetime_string: str
        datetime string in the format 'YYYYMMDD_HHMMSS'

    Returns
    ----------
    datetime.datetime
        corresponding datetime object
    """
    return datetime.datetime(
        int(datetime_string[0:4]), 
        int(datetime_string[4:6]),
        int(datetime_string[6:8]),
        int(datetime_string[8:10]),
        int(datetime_string[10:12]),
        int(datetime_string[12:14])
        )


def extract_frame_at_clock_time(
        video_file: str = None, 
        target_time_string: str = None, 
        output_folder: str = None 
        ) -> bool:
    """
    Extract a single frame from a video file at the specified clock time.

    Parameters
    ----------
    video_file: str
        path to the video file
    target_time_string: str
        target clock time in the format 'YYYYMMDD_HHMMSS'
    output_folder: str
        folder where the extracted frame will be saved

    Returns
    ----------
    success: bool
        True if frame was successfully extracted and saved, False otherwise
    """
    # parse start and end times from video file name
    start_time_string = os.path.basename(video_file).split('_')[3]
    end_time_string = os.path.splitext(os.path.basename(video_file))[0].split('_')[4].split('(')[0]

    # convert datetime strings to datetime objects
    target_time = string_to_datetime(target_time_string)
    start_time = string_to_datetime(start_time_string)
    end_time = string_to_datetime(end_time_string)

    print(f"\nProcessing {video_file}")
    print(f'Detected video time range: {start_time} to {end_time}')

    # Open video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_file}'.")
        return False

    # Determine the video time duration
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Check if the target time is beyond the video coverage
    time_offset = (target_time - start_time).total_seconds()
    if time_offset < 0 or (time_offset > duration):
        print(f"Error: Target time {time_offset:.2f}s is outside video duration ({duration:.2f}s)")
        cap.release()
        return False

    # Otherwise, get the appropriate frame
    frame_number = int(time_offset * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not extract frame at {time_offset:.2f}s")
        cap.release()
        return False
    
    # Determine the camera number (should work for Windows or Mac/Linux)
    nearest_folder = video_file.split(os.sep)[-2]
    video_basename = video_file.split(os.sep)[-1]
    # get the default camera number
    ch_int = int(video_basename.split('ch')[1][0:2].replace('_',''))
    # cameras 1-8: in folder with "1" in it
    if '1' in nearest_folder:
        ch = f"0{ch_int}"
    # cameras 9-16: in folder with "2" in it
    elif '2' in nearest_folder:
        ch_int += 8
        ch = f"0{ch_int}" if ch_int==9 else str(ch_int)
    # cameras 17-24: in folder with "3" in it
    elif '3' in nearest_folder:
        ch_int += 16
        ch = str(ch_int)
    else:
        raise ValueError('Could not assign camera numbers.' 
                         'Please place "1", "2", and/or "3" in each respective video folder name to distinguish cameras.')

    # Save to file
    output_image_file = os.path.join(
        output_folder, 
        f"ch{ch}_{target_time_string}.tiff"
        )
    
    # make sure frame has int datatype
    # frame = frame.astype(np.int32)

    if cv2.imwrite(output_image_file, frame):
        print(f"Extracted frame -> {output_image_file}")
        cap.release()
        return True
    else:
        print(f"Failed to save frame")
        cap.release()
        return False
    

def process_video_files(
        video_files: list[str] = None, 
        target_time_string: str = None, 
        output_folder: str = None, 
        ) -> None:
    """
    Extract frames from a list of video files at the specified clock time.

    Parameters
    ----------
    video_files: list[str]
        list of paths to video files
    target_time_string: str
        target clock time in the format 'YYYYMMDD_HHMMSS'
    output_folder: str
        folder where the extracted frames will be saved

    Returns
    ----------
    None
    """
    os.makedirs(output_folder, exist_ok=True)

    print('Target time:', string_to_datetime(target_time_string))

    # Iterate over video files
    for video_file in video_files:
        extract_frame_at_clock_time(video_file, target_time_string, output_folder)
    return


class GCPSelector:
    def __init__(self, img1_file, img2_file, gcp1, zoom=100):
        # Load grayscale images
        self.img1_file = img1_file
        self.img1 = plt.imread(img1_file)
        if self.img1.ndim == 3:
            self.img1 = np.mean(self.img1, axis=2)
        self.img2_file = img2_file
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

        self.fig.suptitle(os.path.basename(self.img2_file))

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

    def get_results(self, out_file=None):
        """Return/save GCP2 with updated pixel coordinates, with skipped GCPs as NaN"""
        gcp2 = self.gcp2
        gcp2 = gcp2.dropna().reset_index(drop=True)
        # print(f"Compiled {len(gcp2)} updated GCPs")

        if out_file:
            gcp2.to_file(out_file)
            print(f'Updated GCP saved to:\n{out_file}')

        return gcp2


def solve_new_pose(
        gcp_df: gpd.GeoDataFrame = None, 
        K: np.array = None, 
        D: np.array = None, 
        rvec1: np.array = None, 
        tvec1: np.array = None
        ) -> tuple[np.array, np.array]:
    """
    Solve for a new camera pose given updated GCP locations.

    Parameters
    ----------
    gcp_df : gpd.GeoDataFrame
        GeoDataFrame containing GCPs with columns 'X', 'Y', 'Z', 'col_sample', 'row_sample'
    K : np.array
        Camera intrinsic matrix
    D : np.array
        Camera distortion coefficients
    rvec1 : np.array
        Initial rotation vector
    tvec1 : np.array
        Initial translation vector

    Returns
    ----------
    rvec2 : np.array
        Refined rotation vector
    tvec2 : np.array
        Refined translation vector
    """
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


def refine_camera_poses(
        image_files: list[str] = None, 
        init_image_files: list[str] = None, 
        init_cams_file: str = None, 
        init_gcp_file: str = None, 
        out_folder: str = None
        ) -> str:
    """
    Refine camera poses for a set of images using user-selected GCPs.

    Parameters
    ----------
    image_files : list[str]
        List of paths to images to refine
    init_image_files : list[str]
        List of paths to initial images corresponding to image_files
    init_cams_file : str
        Path to CSV file containing initial camera parameters
    init_gcp_file : str
        Path to GPKG file containing initial GCPs
    out_folder : str
        Folder to save refined camera parameters

    Returns
    ----------
    new_cams_file : str
        Path to CSV file containing refined camera parameters
    """

    os.makedirs(out_folder, exist_ok=True)

    # Load initial GCPs
    gcp = gpd.read_file(init_gcp_file, layer='gcp_merged')
    gcp = gcp.dropna().reset_index(drop=True)
    gcp = gcp.loc[gcp['Z'] < 0].reset_index(drop=True)
    # gcp['channel'] = [f"ch0{ch}" if ch < 10 else f"ch{ch}" for ch in gcp['channel']]

    # Load initial camera specs
    init_cams = pd.read_csv(init_cams_file)
    for k in ['K', 'dist', 'K_full', 'rvec', 'tvec']:
        init_cams[k] = init_cams[k].apply(literal_eval)
        
    # Iterate over images
    for i, image_file in enumerate(image_files):
        # Check if output file already exists
        ch = "ch" + os.path.basename(image_file).split('ch')[1][0:2]
        new_cam_file = os.path.join(out_folder, f"{ch}_refined_camera.csv")
        if os.path.exists(new_cam_file):
            print('Refined camera pose already exists in file, skipping.')
            continue

        # Get the original image
        init_image_file = [x for x in init_image_files if ch in os.path.basename(x)][0]

        # Subset the initial GCP
        init_gcp = gcp.loc[gcp['channel']==ch]

        # Get camera intrinsics
        init_cam = init_cams.loc[init_cams['channel']==ch]
        K = np.array(init_cam['K'].values[0])
        dist = np.array(init_cam['dist'].values[0])
        K_full = np.array(init_cam['K_full'].values[0])
        rvec1 = np.array(init_cam['rvec'].values[0])
        tvec1 = np.array(init_cam['tvec'].values[0])

        # Check if updated GCP already exist
        new_gcp_file = os.path.join(out_folder, f"{ch}_updated_GCP.gpkg")
        if not os.path.exists(new_gcp_file):

            # Run interactive GCP selection
            selector = GCPSelector(
                init_image_file, image_file, init_gcp, zoom=100
            )

            # Get the updated GCP
            new_gcp = selector.get_results(out_file=new_gcp_file)
        else:
            new_gcp = gpd.read_file(new_gcp_file)

        # Refine camera pose
        rvec2, tvec2 = solve_new_pose(
            new_gcp, K, dist, rvec1, tvec1
        )

        # Save new camera specs
        new_cam = pd.DataFrame({
            'channel': [ch],
            'K': [json.dumps(K.tolist())],
            'dist': [json.dumps(dist.tolist())],
            'K_full': [json.dumps(K_full.tolist())],
            'rvec': [json.dumps(rvec2.tolist())],
            'tvec': [json.dumps(tvec2.tolist())]
        }, index=[i])
        new_cam.to_csv(new_cam_file, index=False)
        print(f'Refined camera saved to:\n{new_cam_file}')
        
    # Merge refined cameras into one file
    new_cam_files = glob(os.path.join(out_folder, '*refined_camera.csv'))
    new_cams_list = []
    for new_cam_file in new_cam_files:
        new_cams_list += [pd.read_csv(new_cam_file)]
    new_cams = pd.concat(new_cams_list).reset_index(drop=True)
    new_cams_file = os.path.join(out_folder, "refined_cameras_merged.csv")
    new_cams.to_csv(new_cams_file, index=False)
    print(f'Merged refined cameras and saved to:\n{new_cams_file}')

    # Remove intermediary files
    # for new_cam_file in new_cam_files:
    #     os.remove(new_cam_file)

    return new_cams_file


def undistort_image(
        img: np.array = None, 
        K: np.array = None, 
        D: np.array = None, 
        K_full: np.array = None, 
        mask_nodata: bool = True
        ) -> np.array:
    """
    Undistort an image using fisheye model with full FOV intrinsics.

    Parameters
    ----------
    img : np.array
        Input distorted image
    K : np.array
        Camera intrinsic matrix
    D : np.array
        Camera distortion coefficients
    K_full : np.array
        Full FOV camera intrinsic matrix
    mask_nodata : bool
        Whether to mask invalid pixels as NaN

    Returns
    ----------
    img_undistorted : np.array
        Undistorted image
    """
    # Undistort with full FOV
    h,w = img.shape
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_full, (w,h), cv2.CV_32FC1)
    img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Mask invalid pixels
    if mask_nodata:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        mask_undistorted = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST)
        valid_mask = mask_undistorted > 0
        img_undistorted = img_undistorted.astype(np.float32)
        img_undistorted[~valid_mask] = np.nan

    return img_undistorted


def orthorectify(
        image_file: str = None,
        dem_file: str = None,
        K: np.ndarray = None,
        D: np.ndarray = None,
        rvec: np.ndarray = None,
        tvec: np.ndarray = None,
        target_res: float = 0.002,
        max_elevation_above_camera: float = 0.0,
        fov_deg: float = 120.0,
        buffer_size: float = 15.0,
        out_folder: str = None,
        target_datetime: str = None
    ) -> xr.DataArray:
    """
    Generate an orthorectified image from a single camera image and DEM.

    Parameters
    ----------
    image_file : str
        Path to the input image file
    dem_file : str
        Path to the DEM file
    K : np.ndarray
        Camera intrinsic matrix
    D : np.ndarray
        Camera distortion coefficients
    rvec : np.ndarray
        Camera rotation vector
    tvec : np.ndarray
        Camera translation vector
    target_res : float
        Target orthorectified pixel resolution (in DEM units)
    max_elevation_above_camera : float
        Maximum elevation above camera to consider (in DEM units)
    fov_deg : float
        Maximum camera field of view to kee p in degrees
    buffer_size : float
        Buffer size around camera position to clip DEM (in DEM units)
    out_folder : str
        Folder to save the orthorectified image

    Returns
    ----------
    ortho_xr : xr.DataArray
        Orthorectified image as an xarray DataArray
    """

    # --- Load image ---
    img_ds = rio.open(image_file)
    img = img_ds.read().astype(np.float32)
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
    bands, _, _ = img.shape

    # --- Load DEM ---
    dem = rxr.open_rasterio(dem_file).squeeze()
    crs = dem.rio.crs
    dem = xr.where(dem == -9999, np.nan, dem)

    # --- Camera rotation matrix and position ---
    R = cv2.Rodrigues(rvec)[0]
    cam_pos = (-R.T @ tvec).flatten()

    # --- Clip DEM around camera ---
    # OpenCV limits the number of rows and columns during remap (max=32627)
    x_cam, y_cam = cam_pos[0], cam_pos[1]
    dem_clipped = dem.sel(
        x=slice(x_cam - buffer_size, x_cam + buffer_size),
        y=slice(y_cam + buffer_size, y_cam - buffer_size)
    )

    if dem_clipped.size == 0:
        raise ValueError("Clipped DEM area is empty — camera footprint does not intersect DEM.")

    # --- Create dense ortho grid at target resolution ---
    x_min, x_max = float(dem_clipped.x.min()), float(dem_clipped.x.max())
    y_min, y_max = float(dem_clipped.y.min()), float(dem_clipped.y.max())
    Nx = int(np.ceil((x_max - x_min)/target_res)) + 1
    Ny = int(np.ceil((y_max - y_min)/target_res)) + 1
    X_grid = np.linspace(x_min, x_max, Nx, dtype=np.float64)
    Y_grid = np.linspace(y_min, y_max, Ny, dtype=np.float64)
    XX, YY = np.meshgrid(X_grid, Y_grid)

    # Interpolate DEM onto ortho grid
    dem_grid = dem_clipped.interp(x=X_grid, y=Y_grid, method="nearest").values.astype(np.float32)
    ZZ = dem_grid

    # --- Flatten grid and transform to camera coordinates ---
    world_pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    tvec = tvec.reshape(3, 1)
    cam_pts = (R @ world_pts.T + tvec).T

    # --- Mask points behind camera or above max elevation ---
    in_front = cam_pts[:,2] > 0
    below_max = world_pts[:,2] <= cam_pos[2] + max_elevation_above_camera
    half_fov_rad = np.radians(fov_deg / 2)
    inside_fov = np.sqrt(cam_pts[:,0]**2 + cam_pts[:,1]**2) / cam_pts[:,2] <= np.tan(half_fov_rad)
    valid = in_front & below_max & inside_fov

    cam_pts = cam_pts[valid]
    world_pts = world_pts[valid]

    # --- Project points to image coordinates ---
    img_pts, _ = cv2.fisheye.projectPoints(
        cam_pts.reshape(-1,1,3),
        rvec=np.zeros((3,1)),
        tvec=np.zeros((3,1)),
        K=K,
        D=D
    )
    u = img_pts[:,0,0]
    v = img_pts[:,0,1]

    # --- Build dense map for remap ---
    map_x = np.full(XX.shape, np.nan, dtype=np.float32)
    map_y = np.full(XX.shape, np.nan, dtype=np.float32)

    # Map valid world points back to grid indices
    ix = np.round((world_pts[:,0] - x_min) / target_res).astype(int)
    iy = np.round((world_pts[:,1] - y_min) / target_res).astype(int)
    map_x[iy, ix] = u
    map_y[iy, ix] = v

    # --- Remap image ---
    ortho = np.zeros((bands, Ny, Nx), dtype=np.float32)
    for b in range(bands):
        ortho[b] = cv2.remap(
            img[b],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan
        )
        ortho[b][np.isnan(map_x)] = np.nan
        ortho[b][np.isnan(map_y)] = np.nan

    # --- Convert to xarray ---
    ortho_xr = xr.DataArray(
        ortho,
        dims=("band", "y", "x"),
        coords={"x": X_grid, "y": Y_grid}
    )
    ortho_xr = ortho_xr.dropna(how='all', dim='x').dropna(how='all', dim='y')
    ortho_xr = ortho_xr.rio.write_crs(crs, inplace=False)

    # --- Save to TIFF ---
    if out_folder is not None:
        # Define output file name
        os.makedirs(out_folder, exist_ok=True)
        ch = 'ch' + os.path.basename(image_file).split('ch')[1][0:2]
        fname = os.path.join(out_folder, f"{ch}_{target_datetime}_orthoimage.tiff")

        # Convert to int datatype
        ortho_xr = xr.where(np.isnan(ortho_xr), 9999, ortho_xr)
        ortho_xr = ortho_xr.astype(np.uint16)

        # Set properties
        ortho_xr = ortho_xr.rio.write_crs(crs)
        ortho_xr = ortho_xr.rio.write_nodata(9999)

        # Save
        ortho_xr.rio.to_raster(os.path.join(out_folder, fname))
        print(f'Orthoimage saved: {fname}')

    return ortho_xr


def mosaic_orthoimages(
    image_files: list[str] = None,
    closest_cam_map_file: str = None,
    output_folder: str = None,
    chunk_size: int | str = 4096
    ) -> None:
    """
    Mosaic orthorectified images by sampling the closest_cam_map_file.
    
    Parameters
    ----------
    image_files: list[str]
        List of paths to orthoimage files to mosaic
    closest_cam_map_file: str
        Path to closest camera map file (raster with integer values indicating source image index)
    output_folder: str
        Folder to save the output mosaic
    chunk_size: int | str
        Chunk size for dask arrays (int or string like "auto" or 4096)

    Returns
    ----------
    None
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f'Using chunk size {chunk_size} to read and compute')

    # --- Read closest cam map (lazy) ---
    print("Lazily reading closest camera map")
    closest_cam_map = rxr.open_rasterio(closest_cam_map_file, chunks=chunk_size).squeeze()
    crs = closest_cam_map.rio.crs

    # --- Read all images lazily to probe properties ---
    print("Lazily reading all orthoimages to determine properties...")
    datasets_lazy = [rxr.open_rasterio(f, masked=True, chunks=chunk_size) for f in image_files]

    # minimum band count
    min_bands = min(len(ds.band.data) for ds in datasets_lazy)
    num_bands = min_bands
    print(f"Detected minimum number of bands across all images: {num_bands}")

    # select only needed bands (lazy)
    datasets_lazy = [ds.sel(band=slice(0, num_bands)) for ds in datasets_lazy]

    # --- Build target grid ---
    # target resolution (mean of sources)
    res_x = np.mean([abs(ds.rio.resolution()[0]) for ds in datasets_lazy])
    res_y = np.mean([abs(ds.rio.resolution()[1]) for ds in datasets_lazy])
    print(f"Target resolution: {res_x:.3f}, {res_y:.3f} m")

    # bounds & transform from closest_cam_map (you could also union all images)
    bounds = closest_cam_map.rio.bounds()
    width = int((bounds[2] - bounds[0]) / res_x)
    height = int((bounds[3] - bounds[1]) / res_y)
    transform = rio.transform.from_bounds(*bounds, width=width, height=height)

    # build a target grid
    target_grid = xr.DataArray(
        da.full((height, width), 0, dtype=np.uint16, chunks=(chunk_size, chunk_size)),
        dims=("y", "x"),
        coords={
            "y": np.linspace(bounds[3], bounds[1], height),
            "x": np.linspace(bounds[0], bounds[2], width),
        },
    ).rio.write_crs(crs).rio.write_transform(transform)

    # reproject the camera index map to the target grid
    print("Reprojecting closest_cam_map to target grid...")
    closest_cam_map_on_target = closest_cam_map.rio.reproject_match(
        target_grid, resampling=rio.enums.Resampling.nearest
    ).astype(np.int32)

    # --- Build a list of per-image contributions ---
    contributions = []
    print("Reprojecting images and building per-image contributions...")
    for i, ds in enumerate(tqdm(datasets_lazy, desc='Processing images')):
        # select bands if ds has more (safe because we already sliced, but double-check)
        if ds.rio.count > num_bands:
            ds = ds.isel(band=slice(0, num_bands))

        # reproject to target grid (lazy)
        if ds.rio.crs != crs:
            ds = ds.rio.reproject(crs)

        reprojected_ds = ds.rio.reproject_match(target_grid, resampling=rio.enums.Resampling.nearest)

        # Standardize nodata to NaN:
        # prefer ds.rio.nodata if present, but also treat 9999 as nodata sentinel if used
        nodata_val = ds.rio.nodata
        if nodata_val is not None:
            reprojected_ds = reprojected_ds.where(reprojected_ds != nodata_val)
        # also handle explicit 9999 sentinel if your images use it:
        reprojected_ds = reprojected_ds.where(reprojected_ds != 9999)

        # Build a boolean mask aligned to the target grid: True where this image contributes
        mask = (closest_cam_map_on_target == i)

        # Apply mask across band dimension (broadcasting will do the right thing)
        # result has dims (band, y, x) with NaN where mask is False
        contribution = reprojected_ds.where(mask)

        # Ensure coords/dims are exactly (band, y, x)
        # reprojected_ds is typically (band, y, x) already; if not, adjust (keep it consistent)
        contribution = contribution.transpose("band", "y", "x")

        contributions.append(contribution)

        # close the source dataset to free handles
        ds.close()

    if not contributions:
        raise ValueError("No contributions created — check image_files / closest_cam_map content.")

    # --- Combine all contributions ---
    print("Combining per-image contributions into final mosaic...")
    mosaic = functools.reduce(lambda a, b: a.combine_first(b), contributions)

    # --- Finalize nodata metadata and dtype ---
    print("Applying final formatting...")
    mosaic = mosaic.dropna(dim='x', how='all').dropna(dim='y', how='all')
    mosaic = mosaic.fillna(9999).astype(np.uint16)
    mosaic = mosaic.rio.write_nodata(9999)
    mosaic = mosaic.rio.write_crs(crs)
    mosaic = mosaic.rio.write_transform(transform)

    # --- Write to disk (computes lazily) ---
    mosaic_file = os.path.join(output_folder, "orthomosaic.tiff")
    print("Saving orthomosaic...")
    mosaic.rio.to_raster(mosaic_file, compute=True)
    print("Saved orthomosaic:", mosaic_file)

    # Close any open datasets
    for ds in datasets_lazy:
        try:
            ds.close()
        except Exception:
            pass

    return
