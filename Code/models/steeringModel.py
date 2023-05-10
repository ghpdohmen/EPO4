import numpy as np
import matplotlib.pyplot as plt
from libs import normalise_angle

time = 0  # time (initial = 0)
v = 50  # velocity (assumed constant)
phi = 0  # angle of the wheels (initial = 0)
L = 10  # wheelbase distance (assumed constant)

rear_axis = np.array([[v * np.cos(phi) * time], [0]])
front_axis = np.array([[L + (v * np.cos(phi) * time)], [v * np.sin(phi) * time]])

angle_to_x = 0
rate_angle_to_x = (v * np.sin(phi)) / L

phi = np.arange(0, 30, 1)  # angle goes from 0 to 30 degrees

rate = np.zeros(len(phi))

for x in range(len(phi)):
    rate[x] = (v * np.sin(x))*x/L

plt.plot(rate)
plt.show()

def __init__(self, wheelbase: float, delta_time: float=0.05)
    self.delta_time = delta_time
    self.wheelbase = wheelbase


def steering(self, x:float, y:float, theta:float, velocity:float, acceleration: float, steering_angle:float)
    new_velocity = velocity + self.delta_time * acceleration
    angular_velocity = new_velocity*np.tan(steering_angle) / self.wheelbase

    new_x = x + velocity*np.cos(theta)*self.delta_time
    new_y = y + velocity*np.sin(theta)*self.delta_time
    new_theta = normalise_angle(theta + angular_velocity*self.delta_time)

    return new_x, new_y, new_theta, new_velocity