# Soo Locks photogrammetry

Pipeline and utility functions for constructing orthoimages for the Soo Locks project. 

## Correspondence

Rainey Aberle, PhD<br>Email: Rainey.K.Aberle@erdc.dren.mil<br>Research Physical Scientist<br>USACE-ERDC Cold Regions Research and Engineering Laboratory (CRREL)

## Installation

I recommend using [Mamba/Micromamba](https://mamba.readthedocs.io/en/latest/index.html) or [Conda/Miniconda](https://www.anaconda.com/docs/getting-started/anaconda/install) for Python package management. To install the required packages in a new environment, run the following in the command line, replacing the path with your local path to this code repository:

```
cd /path/to/soo_locks_photogrammetry
mamba env create -f environment.yml
```

## Running the pipeline

1. Download video or image files from each camera and place them in a local folder. I recommend the following folder structure for easier running later:

```
.
├── ...
├── Soo_locks_photogrammetry    # Root folder for image processing
│   ├── videos                  # Folder containing your downloaded video files
│   ├── inputs                  # Folder downloaded from RDEDrivePub containing required inputs
│   └── outputs                 # Folder where all outputs will be saved (a new folder will be created here for each job/target datetime)
└── ...
```

2. Download the "inputs" folder from RDEDrivePub. This folder contains the following files required by the pipeline:
- Original calibrated cameras: "original_calibrated_cameras.csv"
- Original raw images folder: "original_images"
- Ground control points (GCP): "GCP_merged_stable.gpkg"
- Digital surface most (DSM) from lidar scan: "lidar_DSM_filled_cropped.tif"
- Closest camera map: "closest_camera_map.tiff"

3. Clone this code repository (or fork, then clone for your own use)

```
git clone https://github.com/RaineyAbe/soo_locks_photogrammetry.git
```

4. Activate the Mamba/Conda environment

```
mamba activate soo_locks
```

5. Run the orthoimage pipeline, replacing the folder names with your local paths. 

```
cd /path/to/soo_locks_photogrammetry

python generate_orthoimage.py \
-target_datetime 20251001171500
-video_folder /path/to/videos \
-inputs_folder /path/to/inputs \
-output_folder /path/to/outputs \
```

You should see folders and files being created in your local outputs folder as the pipeline runs. For the full list of pipeline options, run: `python generate_orthoimage.py --help`


## Steps in the pipeline

1. (If `-video_folder` is specified) Extract image frames from the videos at the user-specified target datetime. 

2. (Optional, but recommended) To account for potential drift of the cameras over time, refine camera positions by manually picking matches between the original and new images. Shifts in match positions are then used to calculate the new camera position w.r.t. the original observations. If it looks like an object has moved or been removed between capture times, skip it!

3. Orthorectify the images by mapping them onto the reference lidar DSM. Refined cameras from step #2 are used if applicable. 

4. Mosaic orthoimages. First, images are sampled at a unified grid at the average spatial resolution of the images. Then, image values are sampled from the closest camera (based on positions detected during lidar scanning) at each image pixel. A plot of the closest camera map can be found in the "inputs" folder. 