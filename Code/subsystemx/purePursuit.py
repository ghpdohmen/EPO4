import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
import robot
from misc import mathFunctions
from misc.robotModeEnum import robotMode
from subsystemx.subsystem import subSystem
from subsystemx.subsystemStateEnum import subSystemState

# code for the pure pursuit algorithm for epo 4

class purePursuit(subSystem):
    start_point = []
    end_point = []

    def __init__(self): #initialises all variables
        
        self.intersec_1 = 0
        self.intersec_2 = 0
        self.targetPoint = 0

        self.wheelbase = robot.Robot.wheelBase
        self.x_location = 0
        self.y_location = 0

        self.lookAheadDistance = 100  # the radius of the circle used to calculate the target point. the look ahead distance.

    def start(self):
        if (robot.Robot.operatingMode == robotMode.Manual) | (robot.Robot.operatingMode == robotMode.EStop):
            self.state = subSystemState.Stopped
        else:
            self.state = subSystemState.Started

        robot.Robot.purePursuitState = self.state

    def intersections(self, _location_x, _location_y, _x1, _y1, _x2, _y2):
        """_summary_
        calculates the target point:
         - draws a line using the start and end points
         - creates a circle of radius lookaheaddistance around kitt
         - takes the intersection of the line and the circle
         - returns array of intersections
        uses the shapely library for these calculations

        Args:
            _location_x (_type_): x_position of kitt
            _location_y (_type_): y_position of kitt
            _x1 (_type_): start position x
            _y1 (_type_): start position y
            _x2 (_type_): end position x
            _y2 (_type_): end position y

        Raises:
            IndexError: if there are no intersection there is an error

        Returns:
            _type_: array of intersection coordinates
        """
        _point = Point(_location_x * 100, _location_y * 100)
        _circle = _point.buffer(self.lookAheadDistance)
        _path = LineString([(_x1, _y1), (_x2, _y2)])
        _intersection = _circle.intersection(_path)

        if len(_intersection.coords) == 2:
            return np.array([(_intersection.coords[0]), (_intersection.coords[1])])
        elif len(_intersection.coords) == 1:
            return np.array([(_intersection.coords[0])])
        else:
            raise IndexError("Seems like there are no intersections?")

    def steeringAngle(self, _x_tp, _y_tp):
        """deterimines the steering angle for kitt based on the target point

        Args:
            _x_tp (_type_): x coordinate of the target point
            _y_tp (_type_): y coordinate of the target point

        Returns:
            _steering_angle: steering in put for KITT between 100 and 200
        """
        _alpha = np.arctan2((_x_tp - self.x_location * 100), (_y_tp - self.y_location * 100))
        #print("Pure pursuit angle: " + str(np.degrees(_alpha)))
        _angle = np.arctan((2 * self.wheelbase * 100 * np.sin(_alpha - np.radians(robot.Robot.robotAngle))) / self.lookAheadDistance)
        #print("steer angle: " + str(np.degrees(_angle)))

        return mathFunctions.angle_to_steer(np.degrees(_angle))

    def update(self):
        # updates the location of KITT
        self.x_location = robot.Robot.xCurrent
        self.y_location = robot.Robot.yCurrent
        
        if (self.state == subSystemState.Running) | (self.state == subSystemState.Started):
            self.state = subSystemState.Running
            self.start_point = robot.Robot.startPos
            self.end_point = robot.Robot.endPos
            
            # uses the intersection function to calculate the two intersections
            self.intersec_1 = self.intersections(self.x_location, self.y_location, self.start_point[0], self.start_point[1], self.end_point[0], self.end_point[1])[0]
            self.intersec_2 = self.intersections(self.x_location, self.y_location, self.start_point[0], self.start_point[1], self.end_point[0], self.end_point[1])[1]

            # chooses the target point which is closer to the end point
            self.targetPoint = mathFunctions.which_one_is_closer(self.intersec_1, self.intersec_2, self.end_point)

            # prints the target point and the steering angle
            print("Targetpoint: " + str(self.targetPoint))  # chosen intersection
            print("Steering angle: " + str(self.steeringAngle(self.targetPoint[0], self.targetPoint[1])))

            # sets the robot direction to the calculated value
            robot.Robot.input_servo = self.steeringAngle(self.targetPoint[0], self.targetPoint[1])

        robot.Robot.purePursuitState = self.state

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.purePursuitState = self.state
