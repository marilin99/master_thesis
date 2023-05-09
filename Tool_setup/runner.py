#!/usr/bin/python3
import os 
import subprocess 

# INSERT ABSOLUTE FOLDER PATH WHERE SEM IMAGES ARE INPUT
SRC_PATH_DM = " "
# INSERT ABSOLUTE FOLDER PATH WHERE CZI IMAGES ARE INPUT
SRC_PATH_BAC = " "

if len(os.listdir(SRC_PATH_DM)) > 0:
    # 2nd variable here is the absolute path to the main.py of the diameter measuring pipeline
    subprocess.run(["python", "absolute-path-to-diameter-measuring-pipeline/main.py"])
    
if len(os.listdir(SRC_PATH_BAC)) > 0:
     # 2nd variable here is the absolute path to the main.py of the bacterial cell analysis pipeline
    subprocess.run(["python", "absolute-path-to-bacterial-cell-analysis-pipeline/main.py"])
    
    
    
    
