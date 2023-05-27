# ooga booga hoe

# inputs:
#    location
#    path

# variables:
#   LookAheadDistance
#   TargetPoint

# outputs:

import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
import robot

# speed = robot.Robot.velocity
wheelbase = robot.Robot.wheelBase
x_location = 5
y_location = 5
start_point = [10, 0]
end_point = [0, 10]
# min_lad =
# max_lad =
# K_constant =
lookAheadDistance = 3


# np.clip(K_constant * speed, min_lad, max_lad)

def targetPoint(location_x, location_y, x1, y1, x2, y2):
    _point = Point(location_x, location_y)
    _circle = _point.buffer(lookAheadDistance)
    _path = LineString([(x1, y1), (x2, y2)])
    _intersection = _circle.intersection(_path)

    return np.array([(_intersection.coords[0]), (_intersection.coords[1])])


def steeringAngle(x_tp, y_tp):
    _alpha = np.arctan2(y_tp, x_tp)
    print(np.degrees(_alpha))
    _angle = np.arctan((2 * wheelbase * np.sin(_alpha)) / lookAheadDistance)
    print(_angle)

    return np.degrees(_angle)


# have to decide which one needs to be prioritised
intersec_1 = targetPoint(x_location, y_location, start_point[0], start_point[1], end_point[0], end_point[1])[0]
intersec_2 = targetPoint(x_location, y_location, start_point[0], start_point[1], end_point[0], end_point[1])[1]

print(intersec_1)  # first intersection
print(intersec_2)  # second intersection
print(steeringAngle(intersec_2[0], intersec_2[1]))

# errors:
# seems to work when going to the right, however it does not work when going to the left
# probably an issue with the radian to degree conversion and in general the use of radian for angles.
