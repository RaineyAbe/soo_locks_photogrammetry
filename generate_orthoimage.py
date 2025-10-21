#! /usr/bin/env python

"""
Pipeline for constructing an orthoimage and/or digital surface model from video files for the Soo Locks project.

Usage: 
----------
python generate_orthoimage.py \
-video_folder <path_to_video_folder> \
-target_datetime <YYYYMMDDHHMMSS> \
-inputs_folder <path_to_inputs_folder> \
-output_folder <path_to_output_folder> \
-refine_cameras <0/1> \
-output_res 0.003 \
-threads "all" \
-generate_dsm <0/1>
"""

import argparse
import os
from glob import glob
import sys
import logging
from datetime import datetime
from tqdm import tqdm
# import the utility functions, assuming the ortho_utils.py file is in the root code directory
import ortho_utils


def getparser():
    parser = argparse.ArgumentParser(description='Wrapper script to pull video frames and generate orthoimage for the Soo Locks project.')
    parser.add_argument('-video_folder', default=None, type=str, help='path to folder containing video files')
    parser.add_argument('-target_datetime', default=None, type=str, help='datetime at which to pull video frames')
    parser.add_argument('-inputs_folder', default=None, type=str, help='path to folder containing standard input files')
    parser.add_argument('-output_folder', default=None, type=str, help='path to folder where all outputs will be saved')
    parser.add_argument('-refine_cameras', default=0, type=int, choices=[0,1], help='whether to refine the pre-optimized cameras')
    parser.add_argument('-output_res', default=0.003, type=float, help='output spatial resolution of the orthoimages.')
    parser.add_argument('-threads', default='all', type=str, help='number of threads to use for parallel processes. Options = number of threads or "all".')
    parser.add_argument('-generate_dsm', default=0, type=int, choices=[0,1], help='whether to generate a digital surface model where images overlap')
    return parser


def main():    
    # --- Parse the user arguments ---
    parser = getparser()
    args = parser.parse_args()
    video_folder = args.video_folder
    target_datetime = args.target_datetime
    inputs_folder = args.inputs_folder
    output_folder = args.output_folder
    refine_cameras = bool(args.refine_cameras)
    output_res = args.output_res
    threads = args.threads
    generate_dsm = bool(args.generate_dsm)

    # --- Define output folders ---
    out_folder = os.path.join(output_folder, 'soo_locks_photogrammetry_' + target_datetime)
    os.makedirs(out_folder, exist_ok=True)
    image_folder = os.path.join(out_folder, 'images')
    undistorted_image_folder = os.path.join(out_folder, 'undistorted_images')
    ortho_folder = os.path.join(out_folder, 'orthoimages')
    # folders for refining cameras
    single_band_image_folder = os.path.join(out_folder, 'single_band_images')
    refined_cam_folder = os.path.join(out_folder, 'refined_cameras')
    refined_cam_full_folder = os.path.join(out_folder, 'refined_cameras_full_FOV')
    # folder for creating DSM
    final_stereo_folder = os.path.join(out_folder, 'final_stereo')

    # --- Set up logging ---
    # Define the timestamped log file name
    dt_now = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = os.path.join(out_folder, f'soo_locks_photogrammetry_{dt_now}.log')
    # Configure logging: writes to file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Custom class to redirect print() output
    class LoggerWriter:
        def __init__(self, level):
            self.level = level
        def write(self, message):
            message = message.strip()
            if message:
                self.level(message)
        def flush(self):
            pass
    # Redirect standard output and errors
    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    # --- Get input files ---
    refdem_file = os.path.join(inputs_folder, 'lidar_DSM_filled_cropped.tif')
    distortion_params_file = os.path.join(inputs_folder, 'initial_undistortion_params.csv')
    camera_folder = os.path.join(inputs_folder, 'calibrated_cameras')
    closest_cam_map_file = os.path.join(inputs_folder, 'closest_camera_map.tiff')
    # Make sure they exist
    if not os.path.exists(inputs_folder):
        raise FileNotFoundError(f'Inputs folder not found: {inputs_folder}')
    if not os.path.exists(refdem_file):
        raise FileNotFoundError(f'Reference DEM file not found: {refdem_file}')
    if not os.path.exists(distortion_params_file):
        raise FileNotFoundError(f'Distortion parameters file not found: {distortion_params_file}')
    if not os.path.exists(camera_folder):
        raise FileNotFoundError(f'Camera folder not found: {camera_folder}')
    if not os.path.exists(closest_cam_map_file):
        raise FileNotFoundError(f'Closest camera map file not found: {closest_cam_map_file}')


    # --- Run the workflow ---
    print('\n------------------------------------------')
    print('--- EXTRACTING FRAMES FROM VIDEO FILES ---')
    print('------------------------------------------\n')
    ortho_utils.process_video_files(
        video_files = sorted(glob(os.path.join(video_folder, '*.avi'))), 
        target_time_string = target_datetime,
        output_folder = image_folder
    )

    print('\n-------------------------------------------')
    print('--- CORRECTING INITIAL IMAGE DISTORTION ---')
    print('-------------------------------------------\n')
    ortho_utils.correct_initial_image_distortion(
        params_file = distortion_params_file, 
        image_files = sorted(glob(os.path.join(image_folder, '*.tiff'))), 
        output_folder = undistorted_image_folder, 
        full_fov=True
    )

    if refine_cameras:
        print('\n------------------------')
        print('--- REFINING CAMERAS ---')
        print('------------------------\n')
        # bundle_adjust only takes single-band images and works much better with the cropped FOV.
        # save single band images
        ortho_utils.save_single_band_images(
            image_files = sorted(glob(os.path.join(image_folder, '*.tiff'))),
            output_folder = single_band_image_folder
        )
        # correct initial distortion, save cropped FOV
        ortho_utils.correct_initial_image_distortion(
            params_file = distortion_params_file,
            image_files = sorted(glob(os.path.join(single_band_image_folder, '*.tiff'))),
            output_folder = undistorted_image_folder,
            full_fov = False
        )
        # refine the cameras
        ortho_utils.refine_cameras(
            image_files = sorted(glob(os.path.join(undistorted_image_folder, '*_cropped_fov.tiff'))),
            camera_files = sorted(glob(os.path.join(camera_folder, '*_cropped_fov.tsai'))),
            refdem_file = refdem_file,
            output_folder = refined_cam_folder,
            threads_string = threads
        )
        # now, adjust them back to the full FOV
        ortho_utils.update_tsai_intrinsics_to_full_fov(
            params_file = distortion_params_file,
            ba_cam_folder = refined_cam_folder,
            output_cam_folder = refined_cam_full_folder
        )

    print('\n------------------------------')
    print('--- ORTHORECTIFYING IMAGES ---')
    print('------------------------------\n')
    if refine_cameras:
        print('Using refined cameras')
        camera_files = sorted(glob(os.path.join(refined_cam_full_folder, '*.tsai')))
    else:
        print('Using pre-calibrated cameras')
        camera_files = sorted(glob(os.path.join(camera_folder, '*_full_fov.tsai')))
    ortho_utils.orthorectify(
        image_files = sorted(glob(os.path.join(undistorted_image_folder, '*.tiff'))), 
        camera_files = camera_files, 
        refdem_file = refdem_file, 
        output_folder = ortho_folder,
        out_res = output_res,
        threads_string = threads
        )

    print('\n------------------------------')
    print('--- MOSAICKING ORTHOIMAGES ---')
    print('------------------------------\n')
    ortho_utils.mosaic_orthoimages(
        image_files = sorted(glob(os.path.join(ortho_folder, '*.tiff'))), 
        closest_cam_map_file = closest_cam_map_file, 
        output_folder = out_folder,
        )
    
    if generate_dsm:
        print('\n----------------------')
        print('--- GENERATING DSM ---')
        print('----------------------\n')
        # Run stereo
        print('\nRunning stereo')
        image_files = [x for x in sorted(glob(os.path.join(ortho_folder, '*.tiff')))
                       if 'mosaic' not in os.path.basename(x)]
        if refine_cameras:
            print('Using refined cameras')
            camera_files = sorted(glob(os.path.join(refined_cam_full_folder, '*.tsai')))
        else:
            print('Using pre-calibrated cameras')
            camera_files = sorted(glob(os.path.join(camera_folder, '*_full_fov.tsai')))
        ortho_utils.run_stereo(
            image_files = image_files,
            camera_files = camera_files,
            output_folder = final_stereo_folder,
            refdem_file = refdem_file
        )

        # Rasterize point clouds
        print('\nRasterizing point clouds')
        ortho_utils.rasterize_point_clouds(
            pc_files = sorted(glob(os.path.join(final_stereo_folder, '*', '*PC.tif'))),
            t_res = output_res * 2,
        )

        # Align DSMs to reference DSM
        ortho_utils.align_dems(
            dem_files = sorted(glob(os.path.join(final_stereo_folder, '*', '*DEM.tif'))),
            refdem_file = refdem_file,
            max_displacement= 100,
            threads_string = threads
        )

        # Mosaic DSMs
        print('\nMosaicking DSMs')
        ortho_utils.mosaic_dems(
            dem_files = sorted(glob(os.path.join(final_stereo_folder, '*', '*-trans_source.tif'))),
            output_file = os.path.join(final_stereo_folder, 'DSM_mosaic.tif'),
            threads_string = threads
        )

    print('\nDone! :)')
    
if __name__ == '__main__':
    main()