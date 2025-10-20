# Soo Locks photogrammetry

Pipeline and utility functions for constructing orthoimages for the Soo Locks project. 

## Correspondence

Rainey Aberle<br>Email: Rainey.K.Aberle@erdc.dren.mil<br>Research Physical Scientist<br>USACE-ERDC Cold Regions Research and Engineering Laboratory (CRREL)

## Installation

1. Clone this code repository locally:

`git clone https://github.com/RaineyAbe/soo_locks_photogrammetry.git`

2. Install the Python environment using [Mamba](https://mamba.readthedocs.io/en/latest/) or [Conda](https://anaconda.org/anaconda/conda). Required packages are listed in the `environment.yml` file. To install directly from the file, run the following in the command line:

`mamba env create -f environment.yml`

3. Install the [Ames Stereo Pipeline v. 3.6.0](https://stereopipeline.readthedocs.io/en/latest/installation.html) (ASP). 

Perhaps the simplest way to install ASP is to download the appropriate precompiled binaries, unzip, and place the folder in your local Applications folder. 

To enable running the commands from anywhere, add the "StereoPipeline-3.6.0*/bin" folder to your `PATH` environment variable, replacing with your actual folder name: 

`export $PATH=$PATH:/Applications/StereoPipeline-3.6.0*/bin`

The command line must often be restarted for this change to take effect. 

3. Download the `inputs` folder with auxilliary pipeline files to your local machine. This folder can be found on the shared RDEPub drive, and contains: 
- Reference DSM: `lidar_DSM_filled_cropped.tif`
- Initial camera and distortion parameters from OpenCV: `initial_undistortion_params.csv`
- Pre-calibrated pinhole camera models for both the cropped and full field of view (FOV) of each camera: `calibrated_cameras/*.tsai`
- Map of closest cameras at each pixel in the model space, used for constructing the orthomosaic: `closest_camera_map.tiff`


## Running the pipeline

Video files must be downloaded from the cameras and placed in a local folder. The full pipeline can then be run using the `generate_orthoimage.py` Python script. An example shell script to run the pipeline is provided in `example_run.sh`. 

Steps in the workflow: 

1. Extract video frames at the specified target time. 

2. Correct initial image distortion using the pre-calibrated OpenCV parameters. 

3. (Optional) Refine cameras using ASP's `stereo` and `bundle_adjust` programs. This step is recommended only if it appears the cameras have moved since calibration. Here, dense feature matches are constructed using stereo preprocessing, which are then used as inputs with the lidar reference DEM for bundle adjustment. 

4. Orthorectify images using ASP's `mapproject` program and the lidar reference DEM. 

5. Mosaic orthoimages. First, images are sampled at a unified grid at the specified output spatial resolution. Then, image values are sampled from the closest camera (based on previously-detected locations during lidar scanning) at each image pixel. A plot of the closest camera map can be found in the `inputs` folder. 

6. (Optional) Generate a DSM using the orthoimages and ASP's `stereo` program. Where images overlap, a point cloud is created. Then, point clouds are rasterized/gridded and mosaicked. The default spatial resolution is the user-specified spatial resolution * 2. 

