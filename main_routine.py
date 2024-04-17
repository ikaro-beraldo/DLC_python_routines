# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:26:03 2023

@author: ikaro
"""

import pandas as pd
import numpy as np
from tkinter import Tk
import csv
import math
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats as sts
from data_handling import *
from maze_modeling import *
from extract_frame import *
from data_processing import *
from plot_functions import *
from data_analysis import *
import cv2
from gif_functions import *


#filename = "F:\\Barnes Maze - Mestrad\\Resultados DLC\\C40_3_G3_D2_T4DLC_resnet50_BM_GPUMar26shuffle2_700000.h5"
dh = Data_handling
filename = dh.select_file(multiple=False)
df = pd.read_hdf(filename)
# Get the coordinates for a specific body part
bpt1 = df.xs('nose', level='bodyparts', axis=1).to_numpy()
conf_threshold = 0.95
std_threshold = 0.5

# Get each hole position
position_each_hole = get_hole_positions(df)

# Get the maze centroid
centroid_coords = centroid_inference(position_each_hole)

# Get the maze coordinates in pixel
maze_info_pixel = maze_recreation(position_each_hole, 10, 5, 60)

# Fix coordinates inconsistencies based on confidence interval
body_part_matrix = fix_frames_confidence(bpt1,conf_threshold)

# Define the trial beginning and end based on confidence interval
a = get_trial_beginning_end_all_bp(body_part_matrix, df, 0.95)
body_part_matrix = get_trial_beginning_end(body_part_matrix, 0.95)

# Get trial latency
latency = get_trial_latency(body_part_matrix, fps=30)

## Fix coordinates inconsistencies based on (x,y) diff standard deviation
# body_part_matrix = fix_frames_diff(body_part_matrix,std_threshold)

# Create a figure and axes
figure, axes = plt.subplots()

# Plot the maze recreation
maze_recreation_plot(axes, body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel, plot_frame=False, title='Nose', show=False)
# Show plots on the selected axes    
colorline(body_part_matrix[:,0], body_part_matrix[:,1], cmap="cool")
plt.show()
#plt.plot(body_part_matrix[:,0],body_part_matrix[:,1])


from data_analysis import *
bp_pos_on_maze = get_bp_position_on_maze(body_part_matrix, position_each_hole, maze_info_pixel, centroid_coords)

bp_pos_on_maze_filtered = filter_bp_pos_on_maze(bp_pos_on_maze, method_used='complete', win=10)

p_errors, s_errors = get_p_s_errors(bp_pos_on_maze_filtered,target=1)[0:2]


total_distance = get_distance(body_part_matrix, maze_info_pixel)[1]
inst_speed, inst_speed_entire, av_speed = get_inst_speed(body_part_matrix, maze_info_pixel, time_win=10, fps=30)

print(inst_speed_entire.max())

strategy = get_the_strategy(bp_pos_on_maze_filtered, target=1)
print(strategy)


quadrant_dict = maze_quadrants(maze_info_pixel, body_part_matrix, centroid_coords, position_each_hole, plot_frame=False, title='Nose', show=True, recreate_maze=True)


bp_pos_on_quadrant = get_bp_position_on_quadrant(body_part_matrix, quadrant_dict, fps=30)
plt.plot(bp_pos_on_quadrant)

#create_gif(body_part_matrix, bp_pos_on_maze, centroid_coords, position_each_hole, maze_info_pixel, inst_speed_entire, quadrant_dict, bp_pos_on_quadrant, plot_frame=False, title='Nose', show=False)

big_plot(body_part_matrix, bp_pos_on_maze_filtered, centroid_coords, position_each_hole, maze_info_pixel, inst_speed_entire, quadrant_dict, bp_pos_on_quadrant)