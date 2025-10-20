#! /usr/bin/env python

"""
Utility functions for generating an orthoimage and/or DSM from video files for the Soo Locks project.

Rainey Aberle
2025
"""

import os
from glob import glob
import subprocess
import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2
import shutil
import ast
import rioxarray as rxr
import xarray as xr
import rasterio as rio
import datetime
# Ignore warnings (rasterio throws a warning whenever an image is not georeferenced. Annoying in this case.)
import warnings
warnings.filterwarnings('ignore')


def run_cmd(
        bin: str = None, 
        args: list = None, **kw
        ) -> str:
    """
    Wrapper for subprocess function to execute bash commands.

    Parameters
    ----------
    bin: str
        command to be excuted (e.g., "mapproject")
    args: list
        arguments to the command as a list
    
    Returns
    ----------
    out: str
        log (stdout) as str if the command executed, error message if the command failed
    """
    binpath = shutil.which(bin)
    call = [binpath,]
    if args is not None: 
        call.extend(args)
    try:
        out = subprocess.run(call,check=True,capture_output=True,encoding='UTF-8').stdout
    except:
        out = f"the command {call} failed to run, see corresponding log"
    return out


def write_log_file(
        log: str = None, 
        output_prefix: str = None
        ) -> str:
    """
    Write a log string to a text file with a timestamped name.

    Parameters
    ----------
    log: str
        log string to be written to file
    output_prefix: str
        prefix for the output log file name

    Returns
    ----------
    log_file: str
        path to the written log file
    """
    # create a string of the current datetime
    now_string = (
        str(datetime.datetime.now())
        .replace('-','')
        .replace(' ','')
        .replace(':','')
        .replace('.','')
    )

    # create output file name
    log_file = output_prefix + '_log_' + now_string + '.txt'

    # write to file
    with open(log_file, 'w') as f:
        f.write(log)
    print('Saved log:', log_file)

    return log_file


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
        output_folder: str = None, 
        output_format: str = 'tiff'
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
    output_format: str
        format to save the extracted frame (e.g., 'tiff', 'png', 'jpg')

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
    
    # Determine the camera number
    ch = os.path.basename(video_file).split('ch')[1][0:2]
    if (frame.shape[1] > 4000) & (ch=='1_'):
        ch = '09'
    elif (frame.shape[1] > 4000):
        ch = str(int(ch[0]) + 8)
    else:
        ch = '0' + ch[0]

    # Save to file
    output_image_file = os.path.join(
        output_folder, 
        f"ch{ch}_{target_time_string}.{output_format}"
        )
    # determine save settings based on output format
    save_params = []
    if output_format == 'png':
        save_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif output_format in ['jpg', 'jpeg']:
        save_params = [cv2.IMWRITE_JPEG_QUALITY, 100]

    if cv2.imwrite(output_image_file, frame, save_params):
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
        output_format: str = 'tiff'
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
    output_format: str
        format to save the extracted frames (e.g., 'tiff', 'png', 'jpg')

    Returns
    ----------
    None
    """
    os.makedirs(output_folder, exist_ok=True)

    print('Target time:', string_to_datetime(target_time_string))

    # Iterate over video files
    for video_file in video_files:
        extract_frame_at_clock_time(video_file, target_time_string, output_folder, output_format)
    return


def correct_initial_image_distortion(
        params_file: str = None, 
        image_files: list[str] = None, 
        output_folder: str = None,
        full_fov: bool = True
        ) -> None:
    """
    Correct initial image distortion using pre-computed distortion parameters.

    Parameters
    ----------
    params_file: str
        path to CSV file containing distortion parameters
    image_files: list[str]
        list of paths to image files to be undistorted
    output_folder: str
        folder where the undistorted images will be saved
    full_fov: bool
        whether to undistort to full field of view

    Returns
    ----------
    None
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load the camera and distortion parameters file
    params = pd.read_csv(params_file)
    params['K'] = params['K'].apply(ast.literal_eval)
    params['K_full'] = params['K_full'].apply(ast.literal_eval)
    params['dist'] = params['dist'].apply(ast.literal_eval)

    # Iterate over image files
    for image_file in image_files:
        # Read image
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        h,w = image.shape[:2]

        # Determine the camera number
        ch = os.path.basename(image_file).split('_')[0][2:]

        # Get the respective distortion parameters
        params_im = params.loc[params['camera']==int(ch)].reset_index().iloc[0]
        K = np.array(params_im['K']).reshape(3,3)
        K_full = np.array(params_im['K_full']).reshape(3,3)
        dist = np.array(params_im['dist']).reshape(-1,1)

        # Undistort
        if full_fov:
            # must do some remapping to maintain no data values
            map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_full, (w, h), cv2.CV_32FC1)
            # apply undistortion to the image
            image_undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
            # create a white mask and remap it the same way to find valid areas
            mask = np.ones((h, w), dtype=np.uint8) * 255
            mask_undistorted = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST)
            # convert mask to boolean
            valid_mask = mask_undistorted > 0
            # Now set invalid pixels to NaN
            image_undistorted_nodata = image_undistorted.astype(np.float32)
            image_undistorted_nodata[~valid_mask] = np.nan
        else:
            image_undistorted = cv2.undistort(image, K, dist, None, K)

        # Save to file
        if full_fov:
            image_undistorted_file = os.path.join(
                output_folder, 
                os.path.splitext(os.path.basename(image_file))[0] + '_undistorted_full_fov.tiff'
                )
        else:
            image_undistorted_file = os.path.join(
                output_folder, 
                os.path.splitext(os.path.basename(image_file))[0] + '_undistorted_cropped_fov.tiff'
                )
        cv2.imwrite(image_undistorted_file, image_undistorted)
        print('\nSaved undistorted image:', image_undistorted_file)
    
    print('\nInitial undistortion complete.')
    return


def determine_threads(
        threads_string: str = "all"
        ) -> int:
    """
    Parse the number of threads to use from a string. 

    Parameters
    ----------
    threads_string: str
        number of threads to use. options: string containing a number, or "all".

    Returns
    ----------
    threads: int
        number of threads as an integer.
    """
    if threads_string=='all':
        threads = os.cpu_count()
    else:
        threads = int(threads_string)
    print(f"Will use up to {threads} threads for each process.")
    return threads


def orthorectify(
        image_files: list[str] = None, 
        camera_files: list[str] = None, 
        refdem_file: str = None, 
        output_folder: str = None,
        nodata_value: str = 'NaN',
        out_res: float = 0.003,
        threads_string: str = 'all'
        ) -> None:
    """
    Orthorectify images using the Ames Stereo Pipeline's mapproject function and a reference DEM.

    Parameters
    ----------
    image_files: list[str]
        list of paths to image files to be orthorectified
    camera_files: list[str]
        list of paths to camera model files corresponding to the images
    refdem_file: str
        path to the reference DEM file
    output_folder: str
        folder where the orthorectified images will be saved
    nodata_value: str
        nodata value to use in the output orthorectified images
    out_res: float
        output pixel resolution in meters
    threads_string: str
        number of threads to use during mapproject
    
    Returns
    ----------
    None
    """
    os.makedirs(output_folder, exist_ok=True)

    threads = determine_threads(threads_string)

    # Iterate over files
    for image_file, cam_file in zip(image_files, camera_files):
        # Define output file name
        image_out_file = os.path.join(output_folder, os.path.basename(image_file))

        # Set up and run command
        print('\nOrthorectifying:', image_file)
        args = [
            '--threads', str(threads),
            '--nodata-value', nodata_value,
            '--tr', str(out_res),
            refdem_file, image_file, cam_file, image_out_file
        ]
        log = run_cmd('mapproject', args)

        # Save log to file
        log_prefix = os.path.join(
            output_folder, 
            os.path.splitext(os.path.basename(image_file))[0] + '_mapproject'
            )
        _ = write_log_file(log, log_prefix)

    print('Done orthorectifying.')
    return


def mosaic_orthoimages(
        image_files: list[str] = None, 
        closest_cam_map_file: str = None, 
        output_folder: str = None
        ) -> None:
    """
    Mosaic orthorectified images based on a closest camera map.

    Parameters
    ----------
    image_files: list[str]
        list of paths to orthorectified image files
    closest_cam_map_file: str
        path to the closest camera map file
    output_folder: str
        folder where the orthomosaic will be saved

    Returns
    ----------
    None
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load the map of closest camera
    print("Reading closest camera map")
    closest_cam_map = rxr.open_rasterio(closest_cam_map_file)
    crs = closest_cam_map.rio.crs

    # Load orthoimages
    print("Reading orthoimages")
    datasets = [rxr.open_rasterio(f, masked=True) for f in image_files]

    # Verify consistent CRS
    for ds in datasets:
        if ds.rio.crs != crs:
            raise ValueError(f"CRS mismatch in {ds.rio.nodata}")

    # Determine number of bands (use from first image)
    num_bands = datasets[0].rio.count
    print(f"Detected {num_bands} band(s) per image")

    # Determine target resolution (average or min pixel size)
    res_x = np.mean([abs(ds.rio.resolution()[0]) for ds in datasets])
    res_y = np.mean([abs(ds.rio.resolution()[1]) for ds in datasets])
    print(f"Using target resolution: {res_x:.3f}, {res_y:.3f}")

    # Determine output bounds and grid
    bounds = closest_cam_map.rio.bounds()
    width = int((bounds[2] - bounds[0]) / res_x)
    height = int((bounds[3] - bounds[1]) / res_y)
    transform = rio.transform.from_bounds(*bounds, width=width, height=height)

    # Create a dummy grid (reference for reprojection)
    dummy_grid = xr.DataArray(
        np.nan*np.zeros((height, width), dtype=np.uint8),
        dims=("y", "x"),
        coords={
            "y": np.linspace(bounds[3], bounds[1], height),
            "x": np.linspace(bounds[0], bounds[2], width),
        },
    ).rio.write_crs(crs).rio.write_transform(transform)

    # Reproject images
    print("Reprojecting images to target grid...")
    reprojected = [
        ds.rio.reproject_match(dummy_grid, resampling=rio.enums.Resampling.nearest)
        for ds in datasets
    ]

    # Stack all reprojected images along a "camera" dimension
    stack = xr.concat(reprojected, dim="camera")

    # Reproject closest_cam_map
    print("Reprojecting closest_cam_map to target grid...")
    closest_cam_map = closest_cam_map.rio.reproject_match(dummy_grid, resampling=rio.enums.Resampling.nearest)

    # Initialize mosaic with NaNs for all bands
    print("Creating mosaic...")
    mosaic_shape = (num_bands, height, width)
    mosaic = xr.DataArray(
        np.full(mosaic_shape, np.nan, dtype=np.float32),
        dims=("band", "y", "x"),
        coords={"band": np.arange(1, num_bands + 1), "y": dummy_grid.y, "x": dummy_grid.x},
    ).rio.write_crs(crs).rio.write_transform(transform)

    # Fill mosaic by selecting pixels based on closest_cam_map
    for i in range(len(stack.camera)):
        mask = closest_cam_map.squeeze() == i
        if num_bands == 1:
            mosaic = xr.where(mask, stack.isel(camera=i)[0], mosaic)
        else:
            for b in range(num_bands):
                mosaic[b] = xr.where(mask, stack.isel(camera=i, band=b), mosaic[b])

    # Save mosaic
    os.makedirs(output_folder, exist_ok=True)
    mosaic_file = os.path.join(output_folder, "orthomosaic.tif")
    mosaic.rio.to_raster(mosaic_file)
    print("Saved orthomosaic:", mosaic_file)
    return


def save_single_band_images(
        image_files: list[str] = None, 
        output_folder: str = None
        ):
    """
    Save single-band versions of multi-band images.

    Parameters
    ----------
    image_files: list[str]
        list of paths to multi-band image files
    output_folder: str
        folder where the single-band images will be saved

    Returns
    ----------
    None
    """
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over image files
    for image_file in tqdm(image_files):
        # convert images to single band
        out_fn = os.path.join(output_folder, os.path.basename(image_file))
        if os.path.exists(out_fn):
            continue
        args = [
            "-b", "1",
            image_file, out_fn
        ]
        log = run_cmd('gdal_translate', args)
        write_log_file(
            log, 
            os.path.join(output_folder, 
                         f"{os.path.splitext(os.path.basename(image_file))[0]}_gdal_translate")
            )

    print('Done saving single-band versions of all images.')
    return


def run_stereo(
    image_files: list[str],
    camera_files: list[str],
    output_folder: str,
    stop_point: int = None,
    refdem_file: str = None,
    threads_string: str = 'all'
    ):
    """
    Run stereo processing on pairs of images using Ames Stereo Pipeline's parallel_stereo.

    Parameters
    ----------
    image_files: list[str]
        list of paths to image files
    camera_files: list[str]
        list of paths to camera model files corresponding to the images
    output_folder: str
        folder where the stereo outputs will be saved
    stop_point: int
        stop point for the stereo processing (if any)
    refdem_file: str
        path to reference DEM, required when image files are mapprojected (orthorectified)
    threads_string: str
        number of threads to use for parallel_stereo

    Returns
    ----------
    None
    """
    os.makedirs(output_folder, exist_ok=True)

    # Determine number of threads to use for parallel_stereo
    threads = determine_threads(threads_string)
    
    # Extract numeric camera IDs (assumes pattern like "*ch01*", "*ch12*")
    def get_cam_num(f):
        base = os.path.basename(f)
        try:
            return int(base.split("ch")[1][:2])
        except Exception:
            raise ValueError(f"Could not parse camera number from filename: {base}")

    # Map image -> camera number
    im_numbers = [get_cam_num(f) for f in image_files]
    print("Detected camera numbers:", im_numbers)

    # Sort by camera number (in case input order isn’t sorted)
    sorted_pairs = sorted(zip(im_numbers, image_files, camera_files), key=lambda x: x[0])
    im_numbers, image_files, camera_files = zip(*sorted_pairs)

    # Define valid groups (1–8 and 9–16)
    groups = []
    group1 = [(n, im, cam) for n, im, cam in zip(im_numbers, image_files, camera_files) if 1 <= n <= 8]
    group2 = [(n, im, cam) for n, im, cam in zip(im_numbers, image_files, camera_files) if 9 <= n <= 16]
    if group1:
        groups.append(group1)
    if group2:
        groups.append(group2)

    total_pairs = 0

    # Iterate over groups
    for g_idx, group in enumerate(groups, start=1):
        print(f"\nProcessing camera group {g_idx} ({group[0][0]}-{group[-1][0]})")
        for (n1, im1, cam1), (n2, im2, cam2) in zip(group[:-1], group[1:]):
            # Skip if cameras are not consecutive
            if n2 != n1 + 1:
                continue

            # Define the output prefix
            pair_prefix = os.path.join(
                output_folder,
                f"{os.path.splitext(os.path.basename(im1))[0]}__{os.path.splitext(os.path.basename(im2))[0]}",
                "run",
            )

            # Construct the arguments
            args = [
                "--threads-singleprocess", str(threads),
                "--threads-multiprocess", str(threads),
                "--nodata-value", "NaN",
                im1, im2,
                cam1, cam2,
                pair_prefix,
            ]
            if stop_point:
                args += ["--stop-point", str(stop_point)]
            if refdem_file:
                args += [refdem_file]

            print(f"Running stereo for ch{n1:02d}-ch{n2:02d}")
            log = run_cmd("parallel_stereo", args)

            # save log to file
            _ = write_log_file(log, pair_prefix)

            total_pairs += 1

    if total_pairs == 0:
        print("No valid stereo pairs found.")
    else:
        print(f"Completed {total_pairs} stereo pairs successfully.")
    return


def refine_cameras(
    image_files: list[str] = None,
    camera_files: list[str] = None,
    refdem_file: str = None,
    output_folder: str = None,
    threads_string: str = 'all'
    ) -> None:
    """
    Refine camera models using bundle adjustment with Ames Stereo Pipeline's parallel_bundle_adjust.

    Parameters
    ----------
    image_files: list[str]
        list of paths to image files
    camera_files: list[str]
        list of paths to camera model files corresponding to the images
    refdem_file: str
        path to the reference DEM file
    output_folder: str
        folder where the refined camera models will be saved
    threads_string: str
        number of threads to use for parallel processes. 

    Returns
    ----------
    None
    """

    os.makedirs(output_folder, exist_ok=True)

    # --- Run stereo preprocessing ---
    print("Running stereo pre-processing to create dense feature matches")
    run_stereo(
        image_files=image_files,
        camera_files=camera_files,
        output_folder=output_folder,
        stop_point=1,  # only run pre-processing
        threads_string=threads_string
    )

    # --- Collect match files ---
    print("Collecting match files")
    match_list = sorted(glob(os.path.join(output_folder, "*", "*.match")))
    if not match_list:
        raise RuntimeError("No .match files found in output folder. See relevant stereo logs.")

    # Helper to extract camera number from filename
    def get_cam_num(fname):
        base = os.path.basename(fname)
        try:
            return int(base.split("ch")[1][:2])
        except Exception:
            raise ValueError(f"Could not parse camera number from: {base}")

    # --- Copy matches into grouped run folders ---
    print("Organizing match files into groups...")
    group_match_files = {1: [], 2: []}

    for match_file in match_list:
        # Extract pair name from path: e.g. ".../img_ch01__img_ch02/run/matchfile"
        pair = os.path.basename(os.path.dirname(match_file))  # e.g. "img_ch01__img_ch02"
        chans = [int(x.split("ch")[1][:2]) for x in pair.split("__") if "ch" in x]

        if not chans:
            print(f"Could not determine cameras for: {match_file}")
            continue

        # Determine group from first channel number
        first_channel = chans[0]
        group = 1 if first_channel < 9 else 2

        # Output file name: group prefix + original pair name
        match_out_file = os.path.join(
            output_folder,
            f"run_group{group}-{pair}.match",
        )
        shutil.copy2(match_file, match_out_file)
        group_match_files[group].append(match_out_file)

    # --- Build group-wise image/camera lists from match files ---
    # Determine number of threads to use
    threads = determine_threads(threads_string)

    print("Building image/camera sets from match files...")
    for group in [1, 2]:
        if not group_match_files[group]:
            print(f"Skipping Group {group} — no match files found.")
            continue

        # Extract all camera numbers that appear in matches for this group
        cams_in_group = set()
        for mf in group_match_files[group]:
            pair = os.path.basename(mf).replace(f"run_group{group}-", "").replace(".match", "")
            chans = [int(x.split("ch")[1][:2]) for x in pair.split("__") if "ch" in x]
            cams_in_group.update(chans)

        cams_in_group = sorted(list(cams_in_group))
        print(f"Group {group} cameras: {cams_in_group}")

        # Subset image/camera files by those numbers
        group_images = []
        group_cameras = []
        for n, im, cam in sorted(zip(
            [get_cam_num(f) for f in image_files],
            image_files,
            camera_files,
        ), key=lambda x: x[0]):
            if n in cams_in_group:
                group_images.append(im)
                group_cameras.append(cam)

        if not group_images:
            print(f"No valid images found for Group {group}")
            continue

        # --- Run bundle adjustment for this group ---
        print(f"\n--- GROUP {group} bundle adjustment: ch{cams_in_group[0]:02d}-ch{cams_in_group[-1]:02d} ---")

        args = [
            "parallel_bundle_adjust",
            "--threads", str(threads),
            "--num-iterations", "2000",
            "--num-passes", "2",
            "--inline-adjustments",
            "--force-reuse-match-files",
            "--heights-from-dem", refdem_file,
            "--heights-from-dem-uncertainty", "0.01",
            "--solve-intrinsics",
            "--intrinsics-to-share", "optical_center,other_intrinsics",
            "--intrinsics-to-float", "all",
            "-o", os.path.join(output_folder, f"run_group{group}"),
        ] + group_images + group_cameras

        log = run_cmd("parallel_bundle_adjust", args)
        _ = write_log_file(log, os.path.join(output_folder, f"run_group{group}"))

    print("\nCamera refinement complete.")
    return


def update_tsai_intrinsics_to_full_fov(
    params_file: str = None,
    ba_cam_folder: str = None,
    output_cam_folder: str = None
    ) -> None:
    """
    Update .tsai camera files with optimized intrinsics mapped to full field of view.

    Parameters
    ----------
    params_file: str
        path to CSV file containing optimized camera parameters
    ba_cam_folder: str
        folder containing the .tsai camera files from bundle adjustment
    output_cam_folder: str
        folder where the updated .tsai camera files will be saved

    Returns
    ----------
    None
    """
    os.makedirs(output_cam_folder, exist_ok=True)

    # Load the camera and distortion parameters file
    params = pd.read_csv(params_file)
    params['K'] = params['K'].apply(ast.literal_eval)
    params['K_full'] = params['K_full'].apply(ast.literal_eval)
    params['dist'] = params['dist'].apply(ast.literal_eval) 

    # Get list of cameras from the bundle_adjust folder
    cam_files = sorted(glob(os.path.join(ba_cam_folder, '*.tsai')))

    # Iterate over rows
    for _, row in tqdm(params.iterrows(), total=len(params)):
        camera = row['camera']
        if camera < 10:
            ch = 0 + str(camera)
        else:
            ch = str(camera)

        # Find matching .tsai camera file
        cam_matches = [x for x in cam_files if ch in os.path.basename(x)]
        if not cam_matches:
            print(f"No camera found for camera {ch}")
            continue
        cam_file = cam_matches[0]

        # Read the optimized camera intrinsics from .tsai
        with open(cam_file, "r") as f:
            cam_lines = [l.strip() for l in f.readlines() if l.strip()]

        fu = fv = cu = cv = None
        for line in cam_lines:
            if line.startswith("fu"):
                fu = float(line.split()[-1])
            elif line.startswith("fv"):
                fv = float(line.split()[-1])
            elif line.startswith("cu"):
                cu = float(line.split()[-1])
            elif line.startswith("cv"):
                cv = float(line.split()[-1])

        if None in (fu, fv, cu, cv):
            print(f"Missing intrinsic values in {cam_file}")
            continue

        # Construct intrinsic matrices
        K_opt = np.array([
            [fu, 0, cu],
            [0, fv, cv],
            [0, 0, 1]
            ])
        K_crop = row["K"].reshape(3, 3)
        K_full = row["K_full"].reshape(3, 3)

        # Calculate transform crop -> full
        H = K_full @ np.linalg.inv(K_crop)

        # Map optimized intrinsics into full-FOV coordinate system
        K_opt_full = H @ K_opt
        K_opt_full /= K_opt_full[2, 2]

        fu_full = K_opt_full[0, 0]
        fv_full = K_opt_full[1, 1]
        cu_full = K_opt_full[0, 2]
        cv_full = K_opt_full[1, 2]

        # Update lines
        updated_lines = []
        for line in cam_lines:
            if line.startswith("fu"):
                updated_lines.append(f"fu = {fu_full}")
            elif line.startswith("fv"):
                updated_lines.append(f"fv = {fv_full}")
            elif line.startswith("cu"):
                updated_lines.append(f"cu = {cu_full}")
            elif line.startswith("cv"):
                updated_lines.append(f"cv = {cv_full}")
            else:
                updated_lines.append(line)

        # Write out new camera file
        cam_out_file = os.path.join(
            output_cam_folder, os.path.basename(cam_file).replace(".tsai", "_full_fov.tsai")
        )
        with open(cam_out_file, "w") as f:
            f.write("\n".join(updated_lines) + "\n")
        print('Saved updated camera with full FOV:', cam_out_file)

    return


def rasterize_point_clouds(
        pc_files: list[str] = None, 
        t_res: float = 0.006,
        threads_string: str = 'all'
        ):
    """
    Rasterize point clouds using Ames Stereo Pipeline's point2dem.

    Parameters
    ----------
    pc_files: list[str]
        list of paths to point cloud files
    t_res: float
        target resolution for the rasterized DEMs
    threads_string: str
        number of threads to use ('all' to use all available cores)

    Returns
    ----------
    None
    """
    print('Rasterizing point clouds')

    # Determine number of threads to use
    threads = determine_threads(threads_string)

    # Iterate over point cloud files
    for pc_file in tqdm(pc_files):
        args = [
            '--threads', str(threads),
            '--tr', str(t_res),
            pc_file
        ]
        log = run_cmd('point2dem', args)
        log_prefix = os.path.splitext(pc_file)[0] + '_point2dem'
        log_file = write_log_file(log, log_prefix)

    print('Rasterization of point clouds complete')
    return


def mosaic_dems(
        dem_files: list[str] = None,
        output_file: str = None,
        threads_string: str = 'all'
    ) -> None:
    """
    Mosaic DEM files using Ames Stereo Pipeline's dem_mosaic.

    Parameters
    ----------
    dem_files: list[str]
        list of paths to DEM files to be mosaicked
    output_file: str
        path to the output mosaicked DEM file
    threads_string: str
        number of threads to use ('all' to use all available cores)
        
    Returns
    ----------
    None
    """

    print('Mosaicking DSMs')
    output_folder = os.path.dirname(output_file)
    os.makedirs(output_folder, exist_ok=True)

    # Determine number of threads to use
    threads = determine_threads(threads_string)

    # Run dem_mosaic
    args = [
        '--threads', str(threads),
        '-o', output_file
    ] + dem_files

    log = run_cmd('dem_mosaic', args)
    log_prefix = os.path.join(output_folder, 'dem_mosaic')
    log_file = write_log_file(log, log_prefix)

    print('DEM mosaicking complete.')
    return
