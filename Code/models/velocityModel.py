#Used for the assignments in the manual

#constants
m =  5.6 #kg
b = 5 # Nm/s Viscous friction coefficient
c = 0.1 # Nm/s Air drag coefficient
faMax = 10 # N
fbMax = 14 # N

#variables
v = 0 # velocity in m/s


#model
fd = b*abs(v) + c*v*v