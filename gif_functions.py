# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:47:55 2023

@author: ikaro
"""
from matplotlib import pyplot as plt
from data_handling import *
from extract_frame import *
from color_functions import *
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from plot_functions import *

def plot_trajectory_gif(axes, body_part_matrix, ref, fps=30):
    # Trajectory itself
    c = colorline(body_part_matrix[:,0], body_part_matrix[:,1], cmap="cool", ax=axes)
    cbar = plt.colorbar(c,ax=axes, ticks=[0, 0.5, 1])
    # Get the max, half and min values of elapsed time
    time_ticks = [str(0), str(round(np.shape(ref)[0]/2/fps,2)), str(round(np.shape(ref)[0]/fps,2))]
    cbar.ax.set_yticklabels(time_ticks)
    cbar.ax.set_ylabel('Time elapsed')    
        

def plot_animal_bp_position_gif(axes, bp_pos_on_maze, ref, fps=30):
    
    #colorline(np.arange(len(bp_pos_on_maze))/fps, bp_pos_on_maze, cmap="cool", linewidth=1.5, ax=axes)
    axes.plot(np.arange(len(bp_pos_on_maze))/fps, bp_pos_on_maze, linewidth=1.5)
    axes.set_xlim((0,len(ref)/fps))
    axes.set_ylim((-2,13))
    axes.set_yticks(np.arange(-1,13,1), ['Out','On','#1','#2','#3','#4','#5','#6','#7','#8','#9','#10','#11','#12'])
    axes.set_xlabel("Seconds")
    axes.set_ylabel("BP position")
    axes.set_title("Body part position on maze")
    #plt.show()
    
def plot_instant_speed_maze_gif(axes, body_part_matrix, inst_speed_entire, centroid_coords, position_each_hole, maze_info_pixel, ref=None):
    
    # Plot the maze recreation
    maze_recreation_plot(axes, body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel, plot_frame=False, title='Nose', show=False)
    # Show plots on the selected axes    
    c = colorline(body_part_matrix[:,0], body_part_matrix[:,1], z=inst_speed_entire/inst_speed_entire.max(), cmap="magma", linewidth=1.5, ax=axes)
    cbar = plt.colorbar(c,ax=axes, ticks=[0, 0.5, 1])
    # Get the max, half and min speeds
    speed_ticks = [str(np.floor(np.min(ref))), str(np.floor(np.max(ref))*0.5), str(np.floor(np.max(ref)))]
    cbar.ax.set_yticklabels(speed_ticks)
    cbar.ax.set_ylabel('cm/s')

def plot_instant_speed_gif(axes, inst_speed_entire, body_part_matrix, fps=30, time_window=10, ref=None):
    
    x = np.linspace(0,len(inst_speed_entire)/fps,len(inst_speed_entire))    # Produce x axis
    #c = colorline(x, inst_speed_entire, z=inst_speed_entire/inst_speed_entire.max(), cmap="magma", linewidth=1.5, ax=axes)
    axes.plot(x, inst_speed_entire, linewidth=1.5)
    axes.set_xlim((0,len(ref)/fps))
    axes.set_ylim((0,100))
    #axes.set_yticks(np.arange(-1,13,1), ['Out','On','#1','#2','#3','#4','#5','#6','#7','#8','#9','#10','#11','#12'])
    axes.set_xlabel("Seconds")
    axes.set_ylabel("cm/s")
    bin_length = time_window/fps
    axes.set_title("Instant speed - "+str(round(bin_length,2)) + " sec bin")


# Function to call the heatmap and plot it
def plot_heatmap_gif(body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel,  ref=None, ax=None):
    
    plot_range = [[centroid_coords[0] - maze_info_pixel["maze_radius_pixels"] - 100, centroid_coords[0] + maze_info_pixel["maze_radius_pixels"] + 100], [centroid_coords[1] - maze_info_pixel["maze_radius_pixels"] - 100, centroid_coords[1] + maze_info_pixel["maze_radius_pixels"] + 100]]
    
    # Get the reference heat map 
    hmap_ref, extent_ref = heatmap(ref[:,0], ref[:,1], 5, plt_range=plot_range)
    # Get the heat map info
    hmap, extent = heatmap(body_part_matrix[:,0], body_part_matrix[:,1], 5, plt_range=plot_range)
    # Get maze info
    alphas = np.clip(Normalize(0, hmap.max(), clip=True)(hmap)*1.5, 0.0, 1.)
    colors = Normalize(0, hmap.max(), clip=True)(hmap)
    colors = cm.magma(colors)
    colors[..., -1] = alphas
    
    if ax is None:
        fig, ax = plt.subplots()
        
    # Plot Maze
    maze_recreation_plot(ax, body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel, plot_frame=False, title='Nose', show=False, invert=False)
    ax.invert_yaxis()
    
    #ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    #ax.imshow(bg)
    h = ax.imshow(np.flip(colors, axis=0), extent=extent, cmap='magma', alpha=1)
    cbar = plt.colorbar(h,ax=ax, ticks=[0, 0.5, 1])
    # Get the max, average and min time spent based on hmap values
    max_value = hmap_ref.max()
    time_spent_ticks = [str(0), str(round(max_value*0.5,2)), str(round(max_value,2))]
    cbar.ax.set_yticklabels(time_spent_ticks)
    cbar.ax.set_ylabel('seconds')
    #plt.scatter(body_part_matrix[:,0], body_part_matrix[:,1])
    
def plot_quadrants_gif(quadrant_dict, fps=30, ax=None):
    
    for ii in range(len(quadrant_dict)):
        # Check if the body part is positioned on the specific
        triangle = quadrant_dict['quadrant'+str(ii+1)]    
        
        # Plot the lines separating the quadrants 
        ax.plot(np.array([triangle[0,0], triangle[2,0]]), np.array([triangle[0,1], triangle[2,1]]), color='black', alpha=0.2)
    
    # Invert the yaxis
    #plt.gca().invert_yaxis()
    #plt.show()
    
def plot_animal_quadrant_position_gif(axes, bp_pos_on_quadrant, ref, fps=30):
    
    #colorline(np.arange(len(bp_pos_on_quadrant))/fps, bp_pos_on_quadrant, cmap="magma", linewidth=1.5, ax=axes)
    axes.plot(np.arange(len(bp_pos_on_quadrant))/fps, bp_pos_on_quadrant, linewidth=1.5)
    axes.set_xlim((0,len(ref)/fps))
    axes.set_ylim((0,5))
    axes.set_yticks(np.arange(1,5,1), ['#1','#2','#3','#4'])
    axes.set_xlabel("Seconds")
    axes.set_ylabel("Quadrant")
    axes.set_title("Body part position by quadrant")
    

# Create a plot (2,3) with the main informations regarding the trial
def big_plot_gif(body_part_matrix, bp_pos_on_maze, centroid_coords, position_each_hole, maze_info_pixel, inst_speed_entire, quadrant_dict, bp_pos_on_quadrant, ref=None, show=True):
    figure, axes = plt.subplots(3,2, figsize=(12.8, 9.6))  # Define the 6 subplots
        
    # 1st PLOT --> the maze and the body part trajectory
    # Recreate the maze
    maze_recreation_plot(axes[0,0], body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel, plot_frame=False, title='Nose', show=False, invert=True)
    # BP trajectory
    plot_trajectory_gif(axes[0,0], body_part_matrix, ref=ref['body_part_matrix'], fps=30)
    
    # 2nd PLOT --> the body part position considering holes and regions
    plot_animal_bp_position_gif(axes[0,1], bp_pos_on_maze, ref=ref['bp_pos_on_maze'])
    
    # 3rd PLOT --> plot the animal instant speed on maze
    plot_instant_speed_maze_gif(axes[1,0], body_part_matrix, inst_speed_entire, centroid_coords, position_each_hole, maze_info_pixel, ref=ref['inst_speed_entire'])

    # 4th PLOT --> plot the animal instant speed on time
    plot_instant_speed_gif(axes[1,1], inst_speed_entire, body_part_matrix, ref=ref['inst_speed_entire'])
    
    # 5th PLOT --> plot the animal spent on a specif part of the maze as a heat map
    plot_heatmap_gif(body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel, ax=axes[2,0], ref=ref['body_part_matrix'])
    plot_quadrants(quadrant_dict, ax=axes[2,0])
    
    # 6th PLOT --> plot the animal position considering each one of the 4 quadrants
    plot_animal_quadrant_position_gif(axes[2,1], bp_pos_on_quadrant, ref=ref['bp_pos_on_maze'], fps=30)

    # Make sure every subplot is tightly adjusted
    plt.tight_layout()
    # Save figure as pdf
    #plt.savefig("C:\\Users\\ikaro\\Downloads\\Figure big plot.pdf", format="pdf", bbox_inches="tight", dpi=300) 
    if show is True:
        plt.show()
    
    return axes

def create_images_for_gif(t, body_part_matrix, bp_pos_on_maze, centroid_coords, position_each_hole, maze_info_pixel, inst_speed_entire, quadrant_dict, bp_pos_on_quadrant, ref=None, plot_frame=False, title='Nose', show=False, fps=30):
    # Show plots on the selected axes  
    idx = np.arange(0,t+1,1)
    axes = big_plot_gif(body_part_matrix[idx,:], bp_pos_on_maze[idx], centroid_coords, position_each_hole, maze_info_pixel, inst_speed_entire[idx], quadrant_dict, bp_pos_on_quadrant[idx], ref=ref, show=False)
    
    # Plot the animal current position
    axes[0,0].plot(body_part_matrix[idx[-1],0],body_part_matrix[idx[-1],1], '.r')
    axes[0,1].plot(idx[-1]/fps,bp_pos_on_maze[idx[-1]], '.r')
    axes[1,0].plot(body_part_matrix[idx[-1],0],body_part_matrix[idx[-1],1], '.r')
    axes[1,1].plot(idx[-1]/fps,inst_speed_entire[idx[-1]], '.r')
    axes[2,0].plot(body_part_matrix[idx[-1],0],body_part_matrix[idx[-1],1], '.r')
    axes[2,1].plot(idx[-1]/fps,bp_pos_on_quadrant[idx[-1]], '.r')
    
        
    plt.savefig(f'./Gifs/img_{t}.png', 
                transparent = False,  
                facecolor = 'white')    
    plt.close()

def create_gif(body_part_matrix, bp_pos_on_maze, centroid_coords, position_each_hole, maze_info_pixel, inst_speed_entire, quadrant_dict, bp_pos_on_quadrant, plot_frame=False, title='Nose', show=False):
    time = np.arange(0, np.shape(body_part_matrix)[0], 1)
    
    #create a reference dict regarding all the temporal series
    ref = dict({'body_part_matrix':body_part_matrix, 'bp_pos_on_maze':bp_pos_on_maze, 'inst_speed_entire':inst_speed_entire, 'bp_pos_on_quadrant':bp_pos_on_quadrant})
    
    for t in time:
        create_images_for_gif(t, body_part_matrix, bp_pos_on_maze, centroid_coords, position_each_hole, maze_info_pixel, inst_speed_entire, quadrant_dict, bp_pos_on_quadrant, ref=ref, plot_frame=False, title='Nose', show=False)
        
    frames = []
    for t in time:
        image = imageio.v2.imread(f'./Gifs/img_{t}.png')
        frames.append(image)
        print(t)

    imageio.mimsave('./Gifs/example1.gif', # output gif
                    frames,          # array of input frames
                    fps = 30)         # optional: frames per second

