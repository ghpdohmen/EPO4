import matplotlib.pyplot
import numpy as np
from matplotlib import pyplot

#Used for the assignments in the manual

#constants
m =  5.6 #kg
b = 5 # Nm/s Viscous friction coefficient
c = 0.1 # Nm/s Air drag coefficient
faMax = 10 # N
fbMax = 14 # N

#variables
vStart = 0 # velocity in m/s
fa = faMax
fb = 0
z = 0

dt = 10 # per 1 ms
timepos = np.zeros(8500)
velocitypos =  np.zeros(8500)

v = vStart
#model
for t in range(0,8500,dt):
    fd = np.sign(v)*(b*np.abs(v) + c*np.power(v,2))
    fres = fa-fb-fd
    a = fres/m
    v += a*(dt/1000)
    z += v*(dt/1000)
    velocitypos[t] = v
    timepos[t] = z

pyplot.figure(1)
pyplot.subplot(211)
pyplot.plot(timepos)
pyplot.subplot(212)
pyplot.plot(velocitypos)
pyplot.show()
#pyplot.savefig("velocitymodeAcceleration")