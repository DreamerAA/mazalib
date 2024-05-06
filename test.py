import numpy as np
import mazalib

side_size = 300
im = np.fromfile('image3d.raw', dtype='uint8')
im = np.reshape(im, (side_size,side_size,side_size))
result = mazalib.unsharp(im, [3.0])
