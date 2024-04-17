# Dump

## Get each hole position
for i in range(12):
    hole = df.xs('h_'+str(i+1), level='bodyparts', axis=1).to_numpy()
    position_each_hole[i,:] = np.array((np.average(hole[:,0]), np.average(hole[:,1])))

## Get the frames where coordinates have > 0.95 confidence
conf_threshold = 0.80
x = bpt1[:,0]
y = bpt1[:,1]
confidence_h = bpt1[:,2] >= 0.95
x_fixed = x
y_fixed = y

exc_frames = np.transpose(np.asarray(np.where(confidence_h == False)))   # Coords excluded (<0.95)
acc_frames = np.transpose(np.asarray(np.where(confidence_h == True)))  # Coords accepted (>0.95)

first_acc = acc_frames[0]  # First accepted frame
last_acc = acc_frames[-1]  # Last accepted frame

# Loop for each excluded frame
for i in range(len(exc_frames)):
    if exc_frames[i] < first_acc:
        x_fixed[exc_frames[i]] = x[first_acc]
        y_fixed[exc_frames[i]] = y[first_acc]
    elif exc_frames[i] > last_acc:
        x_fixed[exc_frames[i]] = x[last_acc]
        y_fixed[exc_frames[i]] = y[last_acc]
    else:
        x_fixed[exc_frames[i]] = x[exc_frames[i]-1]
        y_fixed[exc_frames[i]] = y[exc_frames[i]-1]
        

## Get the frames where coordinates have > 0.5*zscore(abs(diff))
diff_threshold = 0.5
# Get the euclidian distance for each frame
diff_x = np.diff(x)
diff_y = np.diff(y)
diff_euc = sts.zscore(np.sqrt(diff_x**2 + diff_y**2))  # Get the euclidian distance
mask = diff_euc > 0.5                                  # Get which indices

exc_frames = np.transpose(np.asarray(np.where(mask == True)))   # Coords excluded (<0.95)
acc_frames = np.transpose(np.asarray(np.where(mask == False)))  # Coords accepted (>0.95)

first_acc = acc_frames[0]  # First accepted frame
last_acc = acc_frames[-1]  # Last accepted frame

x_fixed2 = x_fixed
y_fixed2 = y_fixed

# Loop for each excluded frame
for i in range(len(exc_frames)):
    if exc_frames[i] < first_acc:
        x_fixed2[exc_frames[i]] = x_fixed[first_acc]
        y_fixed2[exc_frames[i]] = y_fixed[first_acc]
    elif exc_frames[i] > last_acc:
        x_fixed2[exc_frames[i]] = x_fixed[last_acc]
        y_fixed2[exc_frames[i]] = y_fixed[last_acc]
    else:
        x_fixed2[exc_frames[i]] = x_fixed[exc_frames[i]-1]
        y_fixed2[exc_frames[i]] = y_fixed[exc_frames[i]-1]
                    
# Create a figure and axes
figure, axes = plt.subplots()
#print(position_each_hole)
#print(bpt1)
axes.imshow(frame)
print(centroid_coords)
print(maze_info_pixel["hole_radius_pixels"])
maze_circle = plt.Circle((centroid_coords[0],centroid_coords[1]), radius=maze_info_pixel["maze_radius_pixels"], fill=False ) 
 
axes.set_aspect( 1 ) 
axes.add_artist(maze_circle) 
plt.plot(position_each_hole[:,0], position_each_hole[:,1], '.r', markersize=maze_info_pixel["hole_radius_pixels"])
plt.plot(centroid_coords[0], centroid_coords[1], 'og')
plt.plot(x_fixed2,y_fixed2)
plt.show()

#fig1 = plt.figure("Figure 1")
#plt.plot(bpt1[:,2])

#fig2 = plt.figure("Figure 2")
#plt.plot(sts.zscore(abs(np.diff(bpt1[:,0]))))

#print(np.average(sts.zscore(abs(np.diff(bpt1[:,0])))))
#mask_x = sts.zscore(abs(np.diff(bpt1[:,0]))) > 0.5
#plt.plot(mask_x)


################################# PLOT CIRCLE ##########################################
import matplotlib.pyplot as plt 
 
print(maze_info_pixel["pixelcm_ratio"])
figure, axes = plt.subplots() 
cc = plt.Circle((centroid_coords), radius=maze_info_pixel["maze_radius_pixels"], fill=False ) 
 
axes.set_aspect( 1 ) 
axes.add_artist( cc ) 
plt.xlim((-1000, 1000))
plt.ylim((-1000, 1000))
plt.title( 'Colored Circle' ) 
plt.show()

######################################## SPEED ##################################

def get_inst_speed(body_part_matrix, maze_info_pixel, time_win=1, fps=30):
    # Euclidean distance
    euc_distance = np.array(np.sqrt(np.diff(body_part_matrix[:,0])**2 + np.diff(body_part_matrix[:,1])**2))
    # Convert to cm
    euc_distance = euc_distance/maze_info_pixel["pixelcm_ratio"]
    # Add a 0 for the first frame
    euc_distance = np.append([0], euc_distance, axis=0)
    # Time window in sec
    time_win_sec = time_win/fps
    # Calculate speed
    print(np.int64(np.floor(euc_distance.shape[0]/time_win)))
    inst_speed = np.empty((np.int64(np.floor(euc_distance.shape[0]/time_win)),1))
    
    for i in range(len(inst_speed)):
        inst_speed[i] = euc_distance[i]/time_win_sec
        
    # Mean speed
    print(inst_speed)
    av_speed = np.mean(inst_speed)
        
    return inst_speed, av_speed