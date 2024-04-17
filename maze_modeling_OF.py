# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:34:02 2024

@author: ikaro
"""

import numpy as np
from matplotlib import pyplot as plt
from plot_functions import *
from minors_functions import linear_function, generate_points, fit_rectangle_from_multiple_points
import math

def get_vertex_positions(dlc_data_frame, bp_names, prev_vertex_position=None):
    min_confidence = 0.99    # Minimum confidence threshold
    # Create an empty matrix for the vertices coordinates
    position_each_vertex = np.zeros((8,2))
    # Loop for each one the 12 vertices
    for i in range(8):
        # convert the vertex x,y and confidence values to numpy array
        vertex = dlc_data_frame.xs(bp_names['v_'+str(i+1)], level='bodyparts', axis=1).to_numpy()
        # compare the confidence for each frame 
        confidence_mask = np.where(vertex[:,2] >= min_confidence)
        # average the (x,y) coords for only the frames that have the min confidence
        position_each_vertex[i,:] = np.array((np.average(vertex[confidence_mask,0]), np.average(vertex[confidence_mask,1])))
    
    # Check if all vertices got valid coords values, if not, get a prev_vertex_position
    if np.sum(np.isnan(position_each_vertex)) != 0:
        r,c = np.where(np.isnan(position_each_vertex)==True)  # Get the indices where it is NaN
        for i,j in zip(r,c): # Loop for each NaN value
            position_each_vertex[i,j] = prev_vertex_position[i,j]
        
    
    return position_each_vertex

# Function to get points inside the maze (PIM)
def get_points_inside_maze(dlc_data_frame, bp_names, prev_PIM_position=None):
    min_confidence = 0.99    # Minimum confidence threshold
    # Create an empty matrix for the vertices coordinates
    position_each_PIM = np.zeros((4,2))
    # Loop for each one the 12 vertices
    for i in range(np.shape(position_each_PIM)[0]):
        # convert the vertex x,y and confidence values to numpy array
        vertex = dlc_data_frame.xs(bp_names['PIM_'+str(i+1)], level='bodyparts', axis=1).to_numpy()
        # compare the confidence for each frame 
        confidence_mask = np.where(vertex[:,2] >= min_confidence)
        # average the (x,y) coords for only the frames that have the min confidence
        position_each_PIM[i,:] = np.array((np.average(vertex[confidence_mask,0]), np.average(vertex[confidence_mask,1])))
    
    # Check if all vertices got valid coords values, if not, get a prev_vertex_position
    if np.sum(np.isnan(position_each_PIM)) != 0:
        r,c = np.where(np.isnan(position_each_PIM)==True)  # Get the indices where it is NaN
        for i,j in zip(r,c): # Loop for each NaN value
            position_each_PIM[i,j] = prev_PIM_position[i,j]
            
    return position_each_PIM
    

# Calculate the Maze centroid based on the 12 vertices coordinates
def centroid_inference(vertex_coords):
    x = np.sum(vertex_coords[:,0])/len(vertex_coords[:,0])
    y = np.sum(vertex_coords[:,1])/len(vertex_coords[:,1])
    
    return np.array((x,y))

# Recreate the maze area accordingly to real and pixel measurements
def maze_recreation(maze_info_pixels, vertex_coords, PIM_coords):
    # Get the distance between the center of vertexs 1 to 7
    # real_dist_L = 67 # Real distance of the long side in cm
    # real_dist_S = 50 # Real distance of the short side in cm
    
    real_dist_L = maze_info_pixels['box_length'][0] # Real distance of the long side in cm
    real_dist_S = maze_info_pixels['box_length'][1] # Real distance of the short side in cm
    #print('real dist vertexs =' + str(real_dist_vertexs))
    pixel_dist_vertexs_L = np.empty((2,1))
    pixel_dist_vertexs_S = np.empty((2,1))
    
    # Calculate the difference between vertices
    pixel_dist_vertexs_L[0] = np.sqrt((vertex_coords[0,0]-vertex_coords[1,0])**2 + (vertex_coords[0,1]-vertex_coords[1,1])**2) # Pixel distance
    pixel_dist_vertexs_L[1] = np.sqrt((vertex_coords[2,0]-vertex_coords[3,0])**2 + (vertex_coords[2,1]-vertex_coords[3,1])**2) # Pixel distance
    pixel_dist_vertexs_S[0] = np.sqrt((vertex_coords[0,0]-vertex_coords[3,0])**2 + (vertex_coords[0,1]-vertex_coords[3,1])**2) # Pixel distance
    pixel_dist_vertexs_S[1] = np.sqrt((vertex_coords[1,0]-vertex_coords[2,0])**2 + (vertex_coords[1,1]-vertex_coords[2,1])**2) # Pixel distance
    
    # Average it
    pixel_dist_vertexs_L = np.mean(pixel_dist_vertexs_L)
    pixel_dist_vertexs_S = np.mean(pixel_dist_vertexs_S)
    #print('final:' + str(pixel_dist_vertexs))
    
    # Create variables with important parameters
    pixelcm_ratio = np.mean(np.array((pixel_dist_vertexs_L/real_dist_L, pixel_dist_vertexs_S/real_dist_S)))
       
    # Produce a dict with the maze elements PIXEL length values
    #maze_info_pixels = dict({'pixelcm_ratio': pixelcm_ratio})
    maze_info_pixels['pixelcm_ratio'] = pixelcm_ratio
    
    return maze_info_pixels

# Get the x,y coordinates for square vertex inside the OF grid (N list with each square: ((x,y),(x,y),(x,y),(x,y)))
def generate_grid_square_coordinates(rectangle_vertices, x_y_grid_size=(5,5)):
    # The x_y_grid_size must be an array with x: number of squares on x axis and y: number of squares on y axis
    # x_y_grid = (5,5) or (4,3), for example
    # Ensure the rectangle_vertices has exactly 4 vertices
    if rectangle_vertices.shape != (4, 2):
        raise ValueError("Rectangle should have exactly 4 vertices with 2 coordinates each")
    # Sort vertices in clockwise order
    rectangle_vertices = rectangle_vertices[np.argsort(np.arctan2(rectangle_vertices[:, 1], rectangle_vertices[:, 0]))]
    
    # Extract coordinates of the rectangle vertices
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = rectangle_vertices
    
    # Get the lower and upper sides grid line starts (the intersections with the grid vertical lines)
    x_grid_size = x_y_grid_size[0]
    lower = generate_points(x2, y2, x1, y1, x_grid_size)
    upper = generate_points(x4, y4, x3, y3, x_grid_size)
    
    # Get the left and right sides grid line starts (the intersections with the grid horizontal lines)
    y_grid_size = x_y_grid_size[1]
    left = generate_points(x2, y2, x4, y4, y_grid_size)
    right = generate_points(x1, y1, x3, y3, y_grid_size)
    
    # Create a np matrix to store all the x,y coordinates for each grid point
    x_grid_points = np.zeros((x_grid_size+1,y_grid_size+1))
    y_grid_points = np.zeros((x_grid_size+1,y_grid_size+1))
    
    # Loop for each combination of upper and lower rectangle sides
    for i in range(x_grid_size+1):  # Number of lines 
        x_grid_points[:,i] = generate_points(lower[i,0], lower[i,1], upper[i,0], upper[i,1], y_grid_size)[:,0]    # Important to take the y_grid_size
        y_grid_points[:,i] = generate_points(lower[i,0], lower[i,1], upper[i,0], upper[i,1], y_grid_size)[:,1]    # Important to take the y_grid_size
    

    # Get the coordinate for each square
    square_coordinates = list()
    for i in range(x_grid_size): # Number of squares (x axis)
        for j in range(y_grid_size): # Number of squares (y axis)
            # Define the 4 coordinates for each grid square (the vertices of each square)
            square_coordinates.append([[x_grid_points[0+i,0+j], y_grid_points[0+i,0+j]],
                      [x_grid_points[1+i,0+j], y_grid_points[1+i,0+j]],
                      [x_grid_points[1+i,1+j], y_grid_points[1+i,1+j]],
                      [x_grid_points[0+i,1+j], y_grid_points[0+i,1+j]]])
            
    return square_coordinates


def get_region_vertices(square_coordinates, indices):
    # Convert indices to 1-D np.ndarray if they aren't already
    if np.shape(indices) == ():
        indices = np.asarray([indices])
                
    # Check if the array matches the given conditions
    if np.max(indices) >= len(square_coordinates):
        raise ValueError("Invalid indices provided")

    region_vertices = []
    # Group squares in specific regions
    for index in indices:
        square_vertices = square_coordinates[index]
        region_vertices.extend(square_vertices)
    # Convert no np array
    region_vertices = np.asarray(region_vertices)
    # Fit the best rectangle to the polygon given by the squares grouping
    region_vertices = np.asarray(fit_rectangle_from_multiple_points(region_vertices))
                
    # Also get elements as list (JSON serialized data type) 
    region_vertices_list = list()
    for element in range(len(region_vertices)):
        region_vertices_list.append(region_vertices[element].tolist())
    
    
    # Return the vertex coordinates
    return np.asarray(region_vertices), region_vertices_list


# Function to model the maze regions defined by the user (sum of grid square areas)
def model_maze_regions(maze_info_pixel, grid_square_coords, quadrant_info):
    # Create a dict to store the regions coordinates
    maze_regions_dict = {}
    maze_regions_dict_list = {}
    # Loop through each quadrant_info key (to get the regions defined by the user)
    for dict_key in quadrant_info.keys():
        indices = quadrant_info[dict_key]   # Get the square indices for the key specific region
        # Store the x,y coordinates for each region
        maze_regions_dict[dict_key], maze_regions_dict_list[dict_key] = get_region_vertices(grid_square_coords, np.asarray(indices))
            
    # Return the dict with the x,y coordinates for each stablished region
    return maze_regions_dict, maze_regions_dict_list


# Function to define the maze quadrants
def maze_quadrants(maze_info_pixel, body_part_matrix, centroid_coords, position_each_vertex, position_each_PIM, plot_frame=False, title='Nose', show=True, recreate_maze=False):
    # Get the points between the exterior vertices 5, 6, 7, 8
    
    # Pre-allocate the x and y vectors
    x = np.empty((4,))
    y = np.empty((4,))
    
    # Define a  x limit regarding maze radius + 100 pixel (arbitrary)
    #x_limit = maze_info_pixel['maze_radius_pixels']+100
    
    # Get the position halfway each vertex
    x[0] = (position_each_vertex[4,0] + position_each_vertex[5,0])/2
    y[0] = (position_each_vertex[4,1] + position_each_vertex[5,1])/2
    x[1] = (position_each_vertex[5,0] + position_each_vertex[6,0])/2
    y[1] = (position_each_vertex[5,1] + position_each_vertex[6,1])/2
    x[2] = (position_each_vertex[6,0] + position_each_vertex[7,0])/2
    y[2] = (position_each_vertex[6,1] + position_each_vertex[7,1])/2
    x[3] = (position_each_vertex[7,0] + position_each_vertex[4,0])/2
    y[3] = (position_each_vertex[7,1] + position_each_vertex[4,1])/2

    # Define the quadrants as triangles (each line is a triangle)
    quadrant1 = np.array([[centroid_coords[0], centroid_coords[1]], [x[3],y[3]], [position_each_vertex[4,0],position_each_vertex[4,1]], [x[0],y[0]]])
    quadrant2 = np.array([[centroid_coords[0], centroid_coords[1]], [x[0],y[0]], [position_each_vertex[5,0],position_each_vertex[5,1]], [x[1],y[1]]])
    quadrant3 = np.array([[centroid_coords[0], centroid_coords[1]], [x[1],y[1]], [position_each_vertex[6,0],position_each_vertex[6,1]], [x[2],y[2]]])
    quadrant4 = np.array([[centroid_coords[0], centroid_coords[1]], [x[2],y[2]], [position_each_vertex[7,0],position_each_vertex[7,1]], [x[3],y[3]]])

    # Create a dict containing all the coords for the 4 different quadrants
    quadrant_dict = dict({'quadrant1': quadrant1, 'quadrant2': quadrant2, 'quadrant3': quadrant3, 'quadrant4': quadrant4})
    
    # Create a dict with lists inside (to save as json)
    quadrant_dict_list = dict({'quadrant1': quadrant1.tolist(), 'quadrant2': quadrant2.tolist(), 'quadrant3': quadrant3.tolist(), 'quadrant4': quadrant4.tolist()})
    
    # Recreate an y (on the maze border) based on the x (radius)
    if recreate_maze is True:
        figure, axes = plt.subplots()
        # Recreate the maze
        maze_recreation_plot(axes, body_part_matrix, centroid_coords, position_each_vertex, maze_info_pixel, plot_frame=False, title='Nose', show=False, invert=False)    
        
        # Plot the lines separating the quadrants
        for i in range(4): 
            plt.plot(x[i],y[i],'.r')
            plt.plot(np.array([centroid_coords[0], x_quad[i]]), np.array([centroid_coords[1], y_quad[i]]))
        
        # Invert the yaxis
        plt.gca().invert_yaxis()
        plt.show()
    
    fig, axes = plt.subplots()
    maze_recreation_plot_OLR(axes, centroid_coords, position_each_vertex, maze_info_pixel)
    
    return quadrant_dict, quadrant_dict_list
