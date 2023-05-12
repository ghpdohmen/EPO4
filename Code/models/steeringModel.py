#constants
import math
from math import cos, sin, tan

import matplotlib.pyplot as plt
import numpy as np

l = 0.335 #wheelbase in meter
v = 0.3 #speed in m/s

#variables
robotAngle = 0
steeringAngle = 22 # in degrees
posX = 0
posY = 0
stoptime = 40 # in seconds
dt = 0.001 # in seconds

#implementing a rear wheel bicycle model of kitt
w = 0
vX = 0
vY = 0
timepos = np.zeros((2,stoptime*1000))

for t in range(0,stoptime*1000, math.floor(dt * 1000)):
    robotAngle += math.degrees(w)*dt
    posX += vX*dt
    posY += vY*dt
    vX = v * cos(math.radians(robotAngle))
    vY = v * sin(math.radians(robotAngle))
    r = l / tan(math.radians(steeringAngle)) #goed
    w = v / r
    print("r: " + str(r))
    print("vy: " + str(vY))
    print("robotAngle: " + str(robotAngle))
    timepos[0, t] = posX
    print(str(posX) + " , " + str(posY))
    timepos[1, t] = posY

plt.plot(timepos[0],timepos[1])
ax =plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()