# Soo Locks photogrammetry

Pipeline and utility functions for constructing orthoimages for the Soo Locks project. 

## Correspondence

Rainey Aberle, PhD<br>Email: Rainey.K.Aberle@erdc.dren.mil<br>Research Physical Scientist<br>USACE-ERDC Cold Regions Research and Engineering Laboratory (CRREL)

## Installation

1. Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)

2. In the "Terminal" of Docker Desktop, pull the Docker image for this project:

`docker pull raineyaberle/soo_locks_photogrammetry`

3. Download the _"inputs"_ folder containing auxilliary files to your local machine. This folder can be found on the shared RDEdrivePUB, and contains: 
- _"lidar_DSM_filled_cropped.tif"_: reference DSM constructed via lidar scanning, with no data values filled and cropped to the approximate model space. 
- _"initial_undistortion_params.csv"_: initial camera and distortion parameters from OpenCV.
- _"calibrated_cameras/*.tsai"_: pre-calibrated pinhole camera models for both the cropped and full field of view (FOV) of each camera.
- _"closest_camera_map.tiff"_: map of the closest camera at each pixel in the model space, used for constructing the orthomosaic.

## Running the pipeline

1. Download video files from each camera and place them in a local folder. I recommend the following folder structure for easier running later:

.
├── ...
├── Soo_locks_photogrammetry    # Root folder for image processing
│   ├── videos                  # Folder containing your downloaded video files
│   ├── inputs                  # Folder where results and imagery for individual sites will be saved
│   └── outputs                 # Folder where all outputs from photogrammetry processing will be saved
└── ...


2. In the Docker Terminal, define environment variables where your folders are. For example:

```
VIDEOS_FOLDER=/Users/rdcrlrka/Research/Soo_locks_photogrammetry/videos
INPUTS_FOLDER=/Users/rdcrlrka/Research/Soo_locks_photogrammetry/inputs
OUTPUTS_FOLDER=/Users/rdcrlrka/Research/Soo_locks_photogrammetry/outputs
```

3. Now, we'll start the Docker container with these local folders mounted:

```
docker run --rm -it \
-v $VIDEOS_FOLDER:/app/videos \
-v $INPUTS_FOLDER:/app/inputs \
-v $OUTPUTS_FOLDER:/app/outputs \
raineyaberle/soo_locks_photogrammetry
```
This started a bash shell with all of the code, your local files, and required packages accessible. Your shell should now look something like:

`(base) mambauser@875217d0f060:/app$`

4. Finally, we can run the orthoimage pipeline:

```
python generate_orthoimage.py \
-video_folder /app/videos \         # Keep this the same
-inputs_folder /app/inputs \        # Keep this the same
-output_folder /app/outputs \       # Keep this the same
-target_datetime 20251001171500     # REPLACE with the datetime you want to pull from the images
```

You should see folders and files being saved in your local outputs folder as the pipeline runs. For the full list of pipeline options, run: `python generate_orthoimage.py --help`


## Steps in the pipeline under the hood

1. Extract video frames at the user-specified target time. 

2. Correct initial image distortion using the pre-calibrated OpenCV parameters. 

3. (Optional) Refine cameras using ASP's `stereo` and `bundle_adjust` programs. This step is recommended only if it appears the cameras have moved since calibration. Here, dense feature matches are constructed using stereo preprocessing, which are then used as inputs with the reference DSM for bundle adjustment. 

4. Orthorectify and apply additional corrections to the images using ASP's `mapproject` program and the reference DSM. 

5. Mosaic orthoimages. First, images are sampled at a unified grid at the user-specified or default (0.003 m) spatial resolution. Then, image values are sampled from the closest camera (based on positions detected during lidar scanning) at each image pixel. A plot of the closest camera map can be found in the "inputs" folder. 

6. (Optional) [_Something wrong with the final gridding of this step, use with caution._] Generate a DSM using the orthoimages and ASP's `stereo` program. Where images overlap, a point cloud is created. Then, point clouds are rasterized/gridded and mosaicked. The default spatial resolution is the user-specified spatial resolution * 2. 

