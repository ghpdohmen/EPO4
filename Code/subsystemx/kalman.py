import math

import filterpy.kalman
import numpy as np
import filterpy as fp

import robot
from misc import mathFunctions
from subsystemx.subsystem import subSystem
from subsystemx.subsystemStateEnum import subSystemState


class kalman(subSystem):
    """
    This class implements an unscented Kalman FIlter (UKF) for an accurate estimate of the location of our robot
    The entire model is handled by this class.

    The class takes in the following arguments from the robot class:
    Position measurement from the localization subsystem
    steering and velocity inputs

    The class gives the following outputs:
    location X and Y of the robot
    angle of the robot
    uncertainty of the estimation
    """

    x = [[0], [0], [0], [0], [0]]
    """state matrix (x, y, xdot, ydot, angle)"""

    dt = 0.2 # in seconds

    def __init__(self):
        self.points = filterpy.kalman.MerweScaledSigmaPoints(n=5, alpha=0.001, beta=2, kappa=0)
        self.UKF = filterpy.kalman.UnscentedKalmanFilter(dim_x=5, dim_z=2, fx=self.updateModel(), dt=self.dt, points=self.points) #TODO: figure out how Hx() works

    def start(self):
        self.state = subSystemState.Started
        robot.Robot.kalmanState = self.state

    def update(self):
        self.state = subSystemState.Running
        robot.Robot.kalmanState = self.state
        self.dt = robot.Robot.loopTime

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.kalmanState = self.state

    def updateModel(_x, _dt):
        """
        This functions dives an estimate of the model at the next state.
        @param dt: time delta
        @return: state matrix at t=t+dt
        """
        # get angle for internal use
        _angle = _x[4]

        # calculate velocity, used in several calculations
        _velocity = math.sqrt(math.pow(_x[2], 2) + math.pow(_x[3], 2))

        # calculate the forces on the robot and add them together
        _fa = mathFunctions.motor_to_force(robot.Robot.input_motor)  # TODO: implement braking/reverse?
        _fd = np.sign(_velocity) * (robot.Robot.b * np.abs(_velocity) + robot.Robot.c * np.power(_velocity, 2))
        _fres = _fa - _fd

        # calculate acceleration, based on Newton's second law
        _a = _fres / robot.Robot.mass

        # calculate new vx and vy
        _vX = (_velocity + _a * _dt) * math.cos(math.radians(_angle))
        _vY = (_velocity + _a * _dt) * math.sin(math.radians(_angle))

        # get steering angle
        _steeringAngle = mathFunctions.steer_to_angle(robot.Robot.input_servo)

        # calculate the new robot angle, based upon the steering angle
        if _steeringAngle != 0:
            _r = robot.Robot.wheelBase / math.tan(math.radians(_steeringAngle))  # calculate turning radius
        # print(self.r)
        # print(self.steeringAngle)
        if (_velocity != 0) & (_r != 0):
            _w = _velocity / _r  # calculate angular velocity
        else:
            _w = 0
        _angle = _angle + _w * _dt

        # update to new state
        _xNew = [[0], [0], [0], [0], [0]]
        # positions
        _xNew[0] = _x[0] + _vX * _dt
        _xNew[1] = _x[1] + _vY * _dt
        # velocity
        _xNew[2] = _vX
        _xNew[3] = _vY
        # angle
        _xNew[4] = _angle
        return _xNew
