#ooga booga hoe

#inputs:
#    location
#    path

#variables:
#   LookAheadDistance
#   TargetPoint

#outputs:

import numpy as np
import robot

speed = robot.Robot.velocity
min_lad = 
max_lad = 
K_constant = 
LookAheadDistance = np.clip(K_constant * speed, min_lad, max_lad)

def SteeringAngle():
    _angle = np.arctan((2*wheelbase*np.sin(alpha))/LookAheadDistance)