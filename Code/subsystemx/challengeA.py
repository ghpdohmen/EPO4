#inputs, starting position and destination
#outputs, vector with direction to be taken

import numpy as np
import robot

def dirVector(x1, y1, x2, y2):
    _start = np.array([x1, y1])
    _end = np.array([x2, y2])
    _direction = _end - _start
    _line = 
    
    return np.array([_direction, _line])


print(dirVector(2, 1, 5, 2))