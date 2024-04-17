import subprocess
import datetime
import numpy as np

THREAD_NUM=4

def get_video_info(fileloc) :
    command = ['ffprobe',
               '-v', 'fatal',
               '-show_entries', 'stream=width,height,r_frame_rate,duration',
               '-of', 'default=noprint_wrappers=1:nokey=1',
               fileloc, '-sexagesimal']
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print(err)
    out = out.decode()
    out = out.split('\n')
    return {'file' : fileloc,
            'width': int(out[0]),
            'height' : int(out[1]),
            'fps': float(out[2].split('/')[0])/float(out[2].split('/')[1]),
            'duration' : out[3] }

def get_video_frame_count(fileloc) : # This function is spearated since it is slow.
    command = ['ffprobe',
               '-v', 'fatal',
               '-count_frames',
               '-show_entries', 'stream=nb_read_frames',
               '-of', 'default=noprint_wrappers=1:nokey=1',
               fileloc, '-sexagesimal']
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print(err)
    out = out.splitlines()
    return {'file' : fileloc,
            'frames' : out[0]}

def read_frame(fileloc,frame,fps,num_frame,t_w,t_h) :
    command = ['ffmpeg',
               '-loglevel', 'fatal',
               '-ss', str(datetime.timedelta(seconds=frame/fps)),
               '-i', fileloc,
               #'-vf', '"select=gte(n,%d)"'%(frame),
               '-threads', str(THREAD_NUM),
               '-vf', 'scale=%d:%d'%(t_w,t_h),
               '-vframes', str(num_frame),
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo', '-']
    #print(command)
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;
    video = np.fromstring(out, dtype='uint8').reshape((num_frame,t_h,t_w,3)) #NHWC
    return video

def write_frames(frames,filepath) :
    assert( frames.dtype == 'uint8' )

    n, h, w, c = frames.shape
    pix_fmt = 'rgb24' if c == 3 else 'gray'

    # adopted from http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
    command = [ 'ffmpeg',
                '-loglevel', 'fatal',
                '-y', # (optional) overwrite output file if it exists
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', '%dx%d'%(w,h), # size of one frame
                '-pix_fmt', pix_fmt,
                '-r', '24', # frames per second
                '-i', '-', # The imput comes from a pipe
                '-an', # Tells FFMPEG not to expect any audio
               '-vcodec', 'libx264',
                filepath ]
    ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
    out, err = ffmpeg.communicate(frames.tostring() )

    if(err) : print('error',err)