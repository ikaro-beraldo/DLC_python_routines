# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 18:24:11 2023

@author: ikaro
"""

# Set of routines to open CSV files as PANDAS DATAFRAMES for VIA data analysis

import pandas as pd
import numpy as np
import subprocess
import python_ffmpeg
import re

# Get video duration 
def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


def get_manual_classification(filename, video_folder):
    # Function to open CSV as pandas DF
    # Get the data frame
    df = pd.read_csv(filename,header=1)
    # Get the unique video files list
    file_list = df['file_list'].unique()
    # Exclude the firt two dataframe entries (VIA demo videos)
    file_list = np.delete(file_list, [0, 1])
    
    # Output Dict
    output_dict = {}
       
    # Inside a loop
    for vid in range(len(file_list)):
    
        # Video filename (replace all the special characters)
        video_filename = (video_folder+file_list[vid][2:-2]).replace('ç','c')
        video_filename = video_filename.replace('ã','a')
        
        
        # Get video length
        vid_length = get_length(video_filename)
        # Get video FPS
        vid_fps = python_ffmpeg.get_video_info(video_filename)['fps']
        # Get the number of frames
        n_frames = round(vid_length*vid_fps)
        
        # Create a numpy array with zeros (with the number of video frames)
        manual_class = np.zeros((n_frames,))
        
        beg = df['temporal_segment_start'][df['file_list'] == file_list[vid]].to_numpy()  # Begnning list 
        end = df['temporal_segment_end'][df['file_list'] == file_list[vid]].to_numpy()    # End list 
        classification = df['metadata'][df['file_list'] == file_list[vid]].to_list()      # Classification list
        # Get the classification for each temporal segment
        ii = 0
        cl = np.zeros((len(beg),))
        for element in classification:
            cl[ii] = int(re.findall(r'\d+', element)[0])
            # cl[ii] = [int(i) for i in [*element] if i.isdigit()][0]
            ii = ii + 1
        
        # Add a classification for each video frame
        for i in range(len(beg)):
            manual_class[round(beg[i]*vid_fps)-1:round(end[i]*vid_fps)] = cl[i]
        
        # Add the final classification to the output dictionary
        output_dict[file_list[vid][2:-2]] = manual_class
        
    return output_dict

## Main function
# CSV filename
#filename = "D:\\AnalisesRafa\\OLR\\AnalisesBia\\VideoAnnotation\\Demo-Video Annotation10Dec2023_15h39m21s_export.csv"
# Video_folder
#video_folder = "D:\\AnalisesRafa\\OLR\\AnalisesBia\\BarnesMaze\\"
#output_dict = get_manual_classification(filename, video_folder)