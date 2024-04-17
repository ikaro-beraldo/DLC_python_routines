# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 14:14:39 2023

@author: ikaro (beraldin maneiro)
"""

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename, askopenfilenames
    
# Function to open a simple dialog box
def select_file(multiple = True):
    
    # Get the filetypes
    filetypes = (('H5', '*.h5'),('All files', '*.*'))
    
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    root.call('wm', 'attributes', '.', '-topmost', True)
    
    # Check if it is multiple files
    if multiple is True:
        filename = askopenfilenames(title='Open files', filetypes=filetypes)
    else:   # Unique file
        filename = askopenfilename(title='Open files', filetypes=filetypes)
        print('Selected file: '+filename)
        
    # show an "Open" dialog box and return the path to the selected file    
    for i in range(len(filename)):
        filename[i].replace('/','//')   # Replace / for // because of Windows pathway default
    
    return filename