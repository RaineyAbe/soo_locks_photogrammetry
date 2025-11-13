# Soo Locks photogrammetry

Pipeline and utility functions for constructing orthoimages for the Soo Locks project. 

## Correspondence

Rainey Aberle, PhD<br>Email: Rainey.K.Aberle@erdc.dren.mil<br>Research Physical Scientist<br>USACE-ERDC Cold Regions Research and Engineering Laboratory (CRREL)

## Installation

I recommend using [Mamba/Micromamba](https://mamba.readthedocs.io/en/latest/index.html) or [Conda/Miniconda](https://www.anaconda.com/docs/getting-started/anaconda/install) for package management. To install the required packages in a new environment, run the following in the command line, replacing the path with your local path to this code repository:

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
│   └── outputs                 # Folder where all outputs will be saved
└── ...
```

2. Clone this code repository (or fork, then clone)

```
git clone https://github.com/RaineyAbe/soo_locks_photogrammetry.git
```

3. Activate the computing environment

```
mamba activate soo_locks
```

4. Run the orthoimage pipeline, replacing the folder names with your local paths. 

```
cd /path/to/soo_locks_photogrammetry

python generate_orthoimage.py \
-target_datetime 20251001171500
-video_folder /path/to/videos \
-inputs_folder /path/to/inputs \
-output_folder /path/to/outputs \
```

You should see folders and files being saved in your local outputs folder as the pipeline runs. For the full list of pipeline options, run: `python generate_orthoimage.py --help`


## Steps in the pipeline under the hood

1. (If "-video_folder" is specified) Extract video frames at the user-specified target datetime. 

2. (Optional, but recommended) To account for potential drift of the cameras over time, refine camera positions by manually picking matches between the original and new images. Shifts in match positions are then used to calculate the new camera position w.r.t. the original observations. 

3. Orthorectify the images by mapping them onto the reference lidar DSM. Refined cameras from step #2 are used if applicable. 

4. Mosaic orthoimages. First, images are sampled at a unified grid at the user-specified or default (0.003 m) spatial resolution. Then, image values are sampled from the closest camera (based on positions detected during lidar scanning) at each image pixel. A plot of the closest camera map can be found in the "inputs" folder. 