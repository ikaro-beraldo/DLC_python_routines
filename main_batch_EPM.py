# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 21:10:44 2024

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
from maze_modeling_EPM import *
from extract_frame import *
from data_processing import *
from plot_functions import *
from data_analysis import *
import cv2
from gif_functions import *
import os
import json

def plot_video_obj_exploration(body_part_matrix_nose, body_part_matrix_head, df, centroid_coords, position_each_vertex, maze_info_pixel, obj_exploration):
    
    body_coord = df.xs(bp_names['body'], level='bodyparts', axis=1).to_numpy()  
    fig, axs = plt.subplots(2)
    axs[1].plot(obj_exploration)

    for i in range(len(obj_exploration)):
        axs[0].plot(body_part_matrix_nose[i,0],body_part_matrix_nose[i,1],'.r')
        axs[0].plot(body_part_matrix_head[i,0],body_part_matrix_head[i,1],'.k')
        axs[0].plot(body_coord[i,0],body_coord[i,1],'.k')
        maze_recreation_plot_OLR(axs[0], centroid_coords, position_each_vertex, maze_info_pixel,show=False, plot_frame=False)
        # Plot the bodyparts
       
        idx = np.arange(0,i+1)
        print('Obj:'+ str(obj_exploration[i]))
        # axs[1].plot(i, obj_exploration[i],'b.')
        # axs.imshow(frame[99:449, 399:899, :]) 
        plt.draw()
               
        plot_ball = axs[1].plot(i, obj_exploration[i],'r.')
        plot_ball = plot_ball.pop(0)
        
        # You can add any other plots or visualizations here.
    
        plt.pause(0.001)  # Pause for a short time to display the frame
        plot_ball.remove()
        # axs[1].cla()
        axs[0].cla()

def plot_video_obj_exploration_grab(video_file, obj_exploration):

    import cv2
    import matplotlib.pyplot as plt
    
    # Load the video using OpenCV or imageio
    video_path = video_file[0]
    video = cv2.VideoCapture(video_path)  # For OpenCV
    # video = imageio.get_reader(video_path)  # For imageio
    
    # Create a Matplotlib figure
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    
    ax1.set_position([0.1, 0.35, 0.8, 0.6])  # [left, bottom, width, height]
    ax2.set_position([0.1, 0.1, 0.8, 0.2])
    # plt.delaxes(ax2)
    plt.delaxes(ax3)
    plt.delaxes(ax4)
    plt.delaxes(ax5)
    plt.delaxes(ax6)
    
    ax2.plot(obj_exploration)
    plt.xlim(0,len(obj_exploration))
    
    i = 0; # frame index
    
    while True:
        grabbed = video.grab()  # Grab the next frame without decoding
    
        if not grabbed:  # Break the loop when the video ends
            break
    
        # Read the grabbed frame
        ret, frame = video.retrieve()
    
        if not ret:
            break
    
        # Process the frame if needed (e.g., image processing, object detection)
    
        # Plot the video frame
        ax1.clear()
        ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        # ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[99:449, 399:899, :])  # Convert BGR to RGB

        # Create additional plots on the second subplot (ax2)
        # idx = np.arange(0,i+1)
        print('Obj:'+ str(obj_exploration[i]))
        plot_ball = ax2.plot(i, obj_exploration[i],'r.')
        plot_ball = plot_ball.pop(0)
        
        # You can add any other plots or visualizations here.
    
        plt.pause(0.001)  # Pause for a short time to display the frame
        plot_ball.remove()
        
        i = i+1 # Add to the iterator
    
    # Release the video capture object
    video.release()
    
    # Close the Matplotlib window when you're done
    plt.close()
    

# Body_part names (in the future it will be obtained from a config file)
# bp_names = {'nose': 'snout',
#                     'head': 'head',
#                     'body': 'body',
#                     'v_1': 'upperleftlow',
#                     'v_2': 'upperrightlow',
#                     'v_3': 'lowerrightlow',
#                     'v_4': 'lowerleftlow',                   
#                     'v_5': 'upperlefthigh',
#                     'v_6': 'upperrighthigh',
#                     'v_7': 'lowerrighthigh',
#                     'v_8': 'lowerlefthigh',
#                     'obj_1_center': 'leftobjectcenter',
#                     'obj_1_edge': 'leftobjectedge',
#                     'obj_2_center': 'rightobjectcenter',
#                     'obj_2_edge': 'tightobjectedge'}

bp_names = {'head': 'head',
                    'body': 'body',
                    'tail': 'tail',
                    'v_1': 'center1',
                    'v_2': 'center2',
                    'v_3': 'center3',
                    'v_4': 'center4',                   
                    'v_5': 'left.openarm1',
                    'v_6': 'left.openarm2',
                    'v_7': 'left.closedarm1',
                    'v_8': 'left.closedarm2',
                    'v_9': 'right.openarm1',
                    'v_10': 'right.openarm2',
                    'v_11': 'right.closedarm1',
                    'v_12': 'right.closedarm2'}
                    

quadrant_info = {'center': (0),
                 'left_open': (1),
                 'left_closed': (2),
                 'right_open': (3),
                 'right_closed': (4)}

# SET parameter values
conf_threshold = 0.95
std_threshold = 0.5
fps = 30
prev_vertex_position = np.zeros((8,2))
check_video_fps = False # Uses the video FPS (it is necessary to have the trial video at the se folder)
max_trial_duration = 5 # In minutes
reference_region = 'center'     # Define a reference region (not that important, but it will give you the time ratio spent on the reference region)

# box_length
maze_info_pixel = dict()
maze_info_pixel['arm_length'] = (36,6) # (long-side, short-side in cm)
# object_diameter


# CREATE A DATA FRAME TO ORGANIZE THE RSULTS FOR ALL THE TRIALS
trial_info = pd.DataFrame(columns=['ID','Group','Day', 'Distance', 'Av_speed'])

# STEP 1 --> SELECT THE MULTIPLE FILES
filename = select_file(multiple=True)

# STEP 2 --> LOOP FOR EACH FILE
for it in range(len(filename)):
    
    # STEP 2.1 - CHECK THE VIDEO FPS
    if check_video_fps is True: 
        # Extract the video fps based on the H5 filename
        fps = get_video_fps(filename[it])
                    
    # STEP 3 --> LOAD FILE AND RECREATE THE MAZE
    # Load the file
    df = pd.read_hdf(filename[it])
    # Fix camera shaking
    df = df_fix_camera_shaking(df, bp_reference_str=bp_names['v_1'])
    # Exclude frames outside of maximum trial duration
    df = exclude_f_past_duration(df, trial_duration=max_trial_duration, fps=fps)
    # Get each vertex position
    position_each_vertex = get_vertex_positions(df, bp_names, prev_vertex_position=prev_vertex_position) 
    prev_vertex_position = position_each_vertex # UPDATE THE vertex POSITION IN CASE OF AN ERROR
    # Get the maze centroid
    centroid_coords = centroid_inference(position_each_vertex)    
    # Get the maze coordinates in pixel
    maze_info_pixel = maze_recreation(maze_info_pixel, position_each_vertex)
    # Recreate the maze quadrants
    # quadrant_dict, quadrant_dict_list = maze_quadrants(maze_info_pixel, [], position_each_vertex, quadrant_info, plot_frame=False, title='Nose', show=True, recreate_maze=False)
    # Model the maze regions defined by the user (the area sum of grid squares)
    maze_regions_dict, maze_regions_dict_list = model_maze_regions(position_each_vertex)
    
    # Get the coordinates for a specific body part
    head_coord = df.xs(bp_names['head'], level='bodyparts', axis=1).to_numpy()  
    # Fix coordinates inconsistencies based on confidence interval
    body_part_matrix_head = fix_frames_confidence(head_coord,conf_threshold)  
   
    # STEP 3.2 --> GET THE TRIAL BEGINNING, END AND LATENCY
    # Define the trial beginning and end based on confidence interval
    #body_part_matrix_nose, beg, end = get_trial_beginning_end_all_bp(body_part_matrix_nose, df, 0.95)
    # OLD FUNCTION -->> body_part_matrix_nose = get_trial_beginning_end(body_part_matrix_nose, 0.95)
    # Get trial latency
    #latency = get_trial_latency(body_part_matrix_nose, fps=30)      
    
    # STEP 4 --> CREATE A CODE FOR THE NOSE POSITION ON MAZE
    # Get the nose position on the maze
    #bp_pos_on_maze = get_bp_position_on_maze_OLR(body_part_matrix_nose, maze_info_pixel, centroid_coords, position_each_vertex, fps=fps)
    # Get the head angle
    #head_angle = get_head_angle(body_part_matrix_nose,body_part_matrix_head)
    # Get the object exploration vector (throughout time)
    #obj_exploration, obj_exp_parameters = get_obj_exploration(bp_pos_on_maze,head_angle, maze_info_pixel, body_part_matrix_nose, body_part_matrix_head, centroid_coords, position_each_vertex, fps=fps)
    
    #video_file = select_file(multiple = True)
    #plot_video_obj_exploration_grab(video_file, obj_exploration)
    #plot_video_obj_exploration(body_part_matrix_nose, body_part_matrix_head, df, centroid_coords, position_each_vertex, maze_info_pixel, obj_exploration)
    
    #ratio_1_total = obj_exp_parameters['ratio_1_total']
    #ratio_2_total = obj_exp_parameters['ratio_2_total']
    #time_obj_1 = obj_exp_parameters['time_obj_1']
    #time_obj_2 = obj_exp_parameters['time_obj_2']

    # Filter the nose position
    #bp_pos_on_maze_filtered = filter_bp_pos_on_maze(bp_pos_on_maze, method_used='complete', win=fps)
    
    # STEP 5 --> GET PRIMARY AND SECUNDARY ERRORS and STRATEGY USED
    #p_errors, s_errors = get_p_s_errors(bp_pos_on_maze_filtered,target=1)[0:2]
    #strategy = get_the_strategy(bp_pos_on_maze_filtered, target=1)
    
    # STEP 6.1 --> GET THE BODY CENTRE COORD AND PROCESS IT
    # Get the coordinates for a specific body part
    body_centre_coord = df.xs(bp_names['body'], level='bodyparts', axis=1).to_numpy()  
    # Fix coordinates inconsistencies based on confidence interval
    body_part_matrix_body_centre = fix_frames_confidence(body_centre_coord,conf_threshold)
    
    # # UPDATE the beginning and end for the body_centre as well
    # body_part_matrix_body_centre, beg, end = get_trial_beginning_end_all_bp(body_part_matrix_body_centre, df, 0.95)
    
    # STEP 7 --> GET THE TOTAL DISTANCE, INSTANT SPEED AND AVERAGE SPEED
    total_distance = get_distance(body_part_matrix_body_centre, maze_info_pixel)[1]
    inst_speed, inst_speed_entire, av_speed = get_inst_speed(body_part_matrix_body_centre, maze_info_pixel, time_win=10, fps=fps)
        
    # STEP 7.1 --> BODY PART POSITION ON REGION
    bp_pos_on_region = get_bp_position_on_region_OF(body_part_matrix_body_centre, maze_regions_dict, fps=fps)

    # STEP 7.2 --> RATIO (TIME ON TARGET/ TIME ON OTHER QUADRANTS)
    ratio_reference_others, time_on_each_region = get_time_on_each_maze_region_OF(bp_pos_on_region, quadrant_info.keys(), reference_region=reference_region, fps=fps)    
      
    # STEP 8 --> ORGANIZE DATA SINCE NOT EVERY SINGLE THING IS BAGUNÇA
    # Get the file name (only the base name)
    basename = list(os.path.basename(filename[it]))
    ID = basename[1]
    group = ''.join(basename[5:7])
    day = basename[3]
       
    
    ######### Create a data frame to append to the final dataframe
    data = pd.DataFrame([[ID, group, day, total_distance, av_speed, ratio_reference_others, time_on_each_region]], 
                        columns = ['ID','Group','Day','Distance', 'Av_speed', 'ratio_reference_others', 'time_on_each_region']) 
    # makes index continuous
    trial_info = pd.concat([trial_info, data], ignore_index = True)  
    
    # Trial temporal series
    trial_temp_series = dict({'bp_pos_on_region':bp_pos_on_region.tolist(), 
                              #'bp_pos_on_maze_filtered': bp_pos_on_maze_filtered.tolist(),
                              #'bp_pos_on_quadrant':bp_pos_on_quadrant.tolist(),
                              'body_part_matrix_body_centre':body_part_matrix_body_centre.tolist(),
                              #'body_part_matrix_nose':body_part_matrix_nose.tolist(),
                              #'body_part_matrix_head':body_part_matrix_head.tolist(),
                              'position_each_vertex':position_each_vertex.tolist(),
                              'centroid_coords':centroid_coords.tolist(),
                              'maze_info_pixel':maze_info_pixel,
                              'maze_regions_dict_list':maze_regions_dict_list,
                              'total_distance':total_distance.tolist(),
                              'inst_speed':inst_speed.tolist(),
                              'inst_speed_entire':inst_speed_entire.tolist(),
                              'time_on_each_region':time_on_each_region.tolist()})
    
    # Get the name to save a file
    cut_idx = ''.join(basename).find('DLC')
    save_filename = os.path.dirname(filename[it])+'/'+''.join(basename[0:cut_idx])+'.txt'
    # Save temporal series as a json file
    with open(save_filename, "w", encoding='utf-8') as fp:
        json.dump(trial_temp_series, fp, indent=4)  # encode dict into JSON
        
    ############
   
    # STEP 9 --> PLOT WITH THE MAIN INFORMATIONS
    save_filename_plot = os.path.dirname(filename[it])+'/'+''.join(basename[0:cut_idx])
    trial_name = ''.join(basename[0:cut_idx])
    big_plot_EPM(body_part_matrix_body_centre, centroid_coords, position_each_vertex, maze_info_pixel, inst_speed_entire, maze_regions_dict, bp_pos_on_region, save_filename_plot, trial_name=trial_name, trial_data=data, show=True, fps=fps)
     
    # Lil' print to inform to the user that this specific trial has been analysed
    print(str(it)+' - '+''.join(basename[0:cut_idx]) + str(': OK!'))
    
# STEP 11 --> Save the final dataframe as json
save_filename = os.path.dirname(filename[it])+'/'+'Final_results'+'.h5'
trial_info.to_hdf(save_filename, key='trial_info', mode='w')  

trial_info = pd.read_hdf(save_filename, key='trial_info')  