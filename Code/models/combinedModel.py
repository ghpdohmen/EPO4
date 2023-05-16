import math
from math import cos

import matplotlib.pyplot
import numpy as np
from matplotlib import pyplot, pyplot as plt

#Used for the assignments in the manual

#constants
m =  5.6 #kg
b = 5 # Nm/s Viscous friction coefficient
c = 0.1 # Nm/s Air drag coefficient
faMax = 10 # N
fbMax = 14 # N
robotAngle = 0
steeringAngle = 5 # in degrees
posX = 0
posY = 0
stoptime = 40 # in seconds
dt = 0.001 # in seconds
l = 0.335 #wheelbase in meter

#variables
vStart = 2 # velocity in m/s
fa = faMax
fb = 0
z = 0


timepos = np.zeros((2,stoptime*1000))
velocitypos =  np.zeros(1000*stoptime)

v = vStart
w = 0
vX = vY = 0
#model
for t in range(0,1000*stoptime,math.floor(dt*1000)):
    fd = np.sign(v)*(b*np.abs(v) + c*np.power(v,2))
    fres = fa-fb-fd
    a = fres/m
    v += a*(dt)
    robotAngle += math.degrees(w) * dt
    posX += vX * dt
    posY += vY * dt
    vX = v * cos(math.radians(robotAngle))
    vY = v * math.sin(math.radians(robotAngle))
    r = l / math.tan(math.radians(steeringAngle))  # goed
    w = v / r
    #print("r: " + str(r))
    #print("vy: " + str(vY))
    #print("robotAngle: " + str(robotAngle))
    #print("v: " + str(v))
    timepos[0, t] = posX
    #print(str(posX) + " , " + str(posY))
    timepos[1, t] = posY
    if t >=  4000:
        steeringAngle = -5
    else:
        steeringAngle = 5
    if t > 12000:
        fa = 0
    if t > 16000:
        fb = 3
        steeringAngle = 20
    if t > 22500:
        fb = 1.5
        steeringAngle = 22.5
    if t > 27500:
        fb = 0
        fa = 3
        steeringAngle = -5


plt.plot(timepos[0],timepos[1])
plt.plot(0, r, marker='o', markersize='2', markeredgecolor='red')
ax =plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()


