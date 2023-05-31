import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
import robot
from misc import mathFunctions
from misc.robotModeEnum import robotMode
from subsystemx.subsystem import subSystem
from subsystemx.subsystemStateEnum import subSystemState


class purePursuit(subSystem):

    def __init__(self):
        self.intersec_1 = 0
        self.intersec_2 = 0
        self.targetPoint = 0

        self.wheelbase = robot.Robot.wheelBase
        self.x_location = 240
        self.y_location = 0
        self.start_point = [240, 0]
        self.end_point = [160, 320]
        self.lookAheadDistance = 100  # in cm

    def start(self):
        if(robot.Robot.operatingMode == robotMode.Manual | robotMode.EStop):
            self.state = subSystemState.Stopped
        else:
            self.state = subSystemState.Started

        robot.Robot.purePursuitState = self.state

    def intersections(self, _location_x, _location_y, _x1, _y1, _x2, _y2):
        _point = Point(_location_x, _location_y)
        _circle = _point.buffer(self.lookAheadDistance)
        _path = LineString([(_x1, _y1), (_x2, _y2)])
        _intersection = _circle.intersection(_path)

        return np.array([(_intersection.coords[0]), (_intersection.coords[1])])

    def steeringAngle(self, _x_tp, _y_tp):
        _alpha = np.arctan2((_x_tp - self.x_location), (_y_tp - self.y_location))
        print(np.degrees(_alpha))
        _angle = np.arctan((2 * self.wheelbase * np.sin(_alpha)) / self.lookAheadDistance)
        print(_angle)

        return np.degrees(_angle)

    def update(self):
        if self.state == subSystemState.Running | subSystemState.Started:
            self.state = subSystemState.Running
            self.intersec_1 = self.intersections(self.x_location, self.y_location, self.start_point[0], self.start_point[1], self.end_point[0], self.end_point[1])[0]
            self.intersec_2 = self.intersections(self.x_location, self.y_location, self.start_point[0], self.start_point[1], self.end_point[0], self.end_point[1])[1]

            self.targetPoint = mathFunctions.which_one_is_closer(self.intersec_1, self.intersec_2, self.end_point)

            print(self.targetPoint)  # chosen intersection
            print(self.steeringAngle(self.targetPoint[0], self.targetPoint[1]))
        robot.Robot.purePursuitState = self.state
    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.modelState = self.state
