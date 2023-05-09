#!/usr/bin/python3
import os 
import subprocess 

SRC_PATH_DM = r"G:\Automaatika\SEM_input"
SRC_PATH_BAC = r"G:\Automaatika\CZI_input"

if len(os.listdir(SRC_PATH_DM)) > 0:
    subprocess.run(["python", r"C:\Users\marilinm\Documents\py_scripts\diameter_pipeline_codes\main.py"])
    
if len(os.listdir(SRC_PATH_BAC)) > 0:
    subprocess.run(["python", r"C:\Users\marilinm\Documents\py_scripts\bacteria_pipeline_codes\main.py"])
    
    
    
    