import math

import filterpy.kalman
import numpy as np
import filterpy as fp
from filterpy.common import Q_discrete_white_noise

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

    dt = 0.2  # in seconds, will be set automatically during runtime
    UKF = None

    def __init__(self):
        self.points = filterpy.kalman.MerweScaledSigmaPoints(n=5, alpha=0.001, beta=2, kappa=0)
        self.UKF = filterpy.kalman.UnscentedKalmanFilter(dim_x=5, dim_z=2, fx=self.updateModel, dt=self.dt,
                                                         points=self.points,
                                                         x_mean_fn=self.state_mean, hx=self.hx)  # TODO: figure out how Hx() works

    def start(self):
        self.state = subSystemState.Started
        robot.Robot.kalmanState = self.state
        self.UKF.x = self.x
        self.UKF.x[0] = robot.Robot.startPos[0]
        self.UKF.x[1] = robot.Robot.startPos[1]
        self.UKF.P = np.diag([0.05, 0.05, 0, 0, 1])
        self.UKF.R = np.diag([0.117, 0.153])  # in meters
        self.UKF.Q = Q_discrete_white_noise(dim=5, dt=self.dt, var=0.1**2, block_size=2)

    def update(self):
        self.state = subSystemState.Running
        robot.Robot.kalmanState = self.state
        self.dt = robot.Robot.loopTime
        self.measurement = np.array(robot.Robot.posXLocalization, robot.Robot.posYLocalization)
        self.UKF.predict(dt=self.dt)
        self.UKF.update(dt=self.dt, z=self.measurement)
        robot.Robot.xCurrent = self.UKF.x[0]
        robot.Robot.yCurrent = self.UKF.x[1]
        robot.Robot.uncertaintyX = self.UKF.P[0][0]
        robot.Robot.uncertaintyY = self.UKF.P[1][1]

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.kalmanState = self.state

    def updateModel(self,state, _dt):
        """
        This functions dives an estimate of the model at the next state.
        @param dt: time delta
        @return: state matrix at t=t+dt
        """

        _x = x
        # get angle for internal use
        _angle = state[4]

        # calculate velocity, used in several calculations
        _velocity = math.sqrt(math.pow(state[2], 2) + math.pow(state[3], 2))

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
        _xNew[0] = state[0] + _vX * _dt
        _xNew[1] = state[1] + _vY * _dt
        # velocity
        _xNew[2] = _vX
        _xNew[3] = _vY
        # angle
        _xNew[4] = _angle
        return _xNew


    def state_mean(_sigmas, _Wm):
        """
        Used in kalman filter to calculate the state mean. Needed because angles can't be added properly
        @param _Wm:
        @return:
        """
        x = np.zeros(3)
        sum_sin, sum_cos = 0., 0.

        for i in range(len(_sigmas)):
            s = _sigmas[i]
            x[0] += s[0] * _Wm[i]
            x[1] += s[1] * _Wm[i]
            x[2] += s[2] * _Wm[i]
            x[3] += s[3] * _Wm[i]
            sum_sin += math.sin(math.radians(s[2])) * _Wm[i]
            sum_cos += math.cos(math.radians(s[2])) * _Wm[i]
        x[4] = math.atan2(sum_sin, sum_cos)
        return x

    def hx (self,x):
        return np.array([x[0],x[1]])
