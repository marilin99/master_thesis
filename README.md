# "Fibar": a Tool for Automated Analysis of Complex Biomaterials from Microscopy Images

This repo covers most of the work done for creating the image processing tool "Fibar". The `dev` branch is for development, while the `main` branch is for users who would like to setup the tool or test some parts out. 

## Project Objective
The goal of this project is to provide an automated microscopy image analysis tool for complex biomaterials. These are samples of electrospun fibers with encapsulated components, such as bacteria, drugs etc. The tool is divided into two pipelines - one for fiber diameter measuring, the other for bacterial cell analysis. The ultimate goal of this tool is to reduce the workload of the researcher by providing a viable alternative to the manual analysis.

## The repo structure 
The `main` branch of the repo includes 3 main folders: `Tool setup`, `Example data` and `Quick runs`. 

The `Tool setup` folder can be used when the user wants to setup the whole system. This system is being triggered by the Task Scheduler in Windows OS - a .xml is provided so the user can setup the task themselves. Additionally, some folder paths ought to be changed in the <code>runner.py</code> and the <code>main.py</code> files in the respective pipelines. The specific lines are marked with comments.

The `Example data` folder includes both real microscopy data for testing out both pipelines.

The `Quick runs` folder is just a more direct way for the reader to test out some pipelines with the example data provided. In separate subfolders, the `main.py` can be run using the command line. In case of the Diameter measuring pipeline, 3 arguments should be provided by the user: absolute file path, name of method, number of diameter measurements (<code>python3 quick_run_dm_pipeline.py abs_file_path_here.tif "U-Net" 10 </code>) The user should be reminded about the constraints of the pipelines, which are provided in the thesis as well as [below](#constraints-of-"Fibar"). 

## Requirements 
The pipelines should be executed in the Python 3.8.10 environment. All of the required libraries and their versions can
be installed from the <code>requirements.txt</code> file.

## Constraints of "Fibar"

For the fiber analysis pipeline, the system assumes the following:
- the SEM input image can be TIF/PNG/JPG format;
- the file name should include "_2k_" or "_5k_" if a magnification of 2k (2000x) or
5k (5000x) was used;
- the OCR part of the pipeline assumes that the value and unit are provided in white
highlighted in black and the horizontal bar is white on a transparent background
aligned below the value and unit;
- the scale can have values 1, 2, 3, 10, 200 or 400 and units, nm or um.

In case of the bacterial analysis pipeline, the following is expected:
- the input file is CZI format;
- the user has used red and green detectors to collect the data from the sample;
- staining dyes or fluorescent proteins have been used in the experiment (that have
resulted in filled fluorescent bacteria);
- the shape of the raw data has the shape (I,T,C,Y,X) where I (illumination) and T
(time) have values of 0;
- the input file can have up to 4 channels, but it needs to includes at least 2 channels
for the red and green color.

