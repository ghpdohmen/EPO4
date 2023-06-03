import matplotlib.pyplot
import numpy as np
from matplotlib import pyplot

#Used for the assignments in the manual
#v156 = -0.2503*exp(-0.2337*t)+0.2888
#v160 = -1.473*exp(-0.3828*t)+1.485
#v165 = -8.266*exp(-1,030*t)+1.645

#constants
m =  5.6 #kg
b = 3.2 # Nm/s Viscous friction coefficient
c = 0.09 # Nm/s Air drag coefficient
faMax = 3 # N
fbMax = 4 # N

#variables
vStart = 1.09 # velocity in m/s
fa = 0
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
    print(z)
    timepos[t] = z

pyplot.figure(1)
pyplot.subplot(211)
pyplot.plot(timepos)
pyplot.subplot(212)
pyplot.plot(velocitypos)
pyplot.show()
#pyplot.savefig("velocitymodeAcceleration")