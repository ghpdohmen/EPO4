import math

import filterpy.kalman
import numpy as np

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

    x = np.array([0, 0, 0, 0, 0])
    """state matrix (x, y, xdot, ydot, angle)"""

    dt = 0.2  # in seconds, will be set automatically during runtime
    UKF = None

    # random caching variables
    vx_old = 0

    def __init__(self):
        self.points = filterpy.kalman.MerweScaledSigmaPoints(n=5, alpha=0.001, beta=2, kappa=1)
        self.UKF = filterpy.kalman.UnscentedKalmanFilter(dim_x=5, dim_z=2, fx=self.updateModel, dt=self.dt,
                                                         points=self.points, hx=self.hx)

    def start(self):
        self.state = subSystemState.Started
        robot.Robot.kalmanState = self.state
        self.UKF.x = self.x
        self.UKF.x = np.array([robot.Robot.startPos[0] / 100, robot.Robot.startPos[1] / 100, 0, 0, 0])
        # self.UKF.P = np.diag([0.05, 0.05, 0.01, 0.01, 1])
        self.UKF.P *= 0.01  # TODO: kijken naar invloed van dit
        print("P: " + str(self.UKF.P))
        # print(str(self.UKF.x))
        self.UKF.R = np.diag([0.117, 0.153])  # in meters
        self.UKF.Q = np.diag([0.1, 0.1, 0.01, 0.01, 1])
        print("Location Kalman start: ( " + str(self.UKF.x[0]) + " , " + str(self.UKF.x[1]) + " ) m")
        print("Velocity Kalman start: ( " + str(self.UKF.x[2]) + " , " + str(self.UKF.x[3]) + " ) m/s")

    def update(self):
        if (robot.Robot.runTime != 0):
            self.measurement = np.array([robot.Robot.posXLocalization, robot.Robot.posYLocalization])
            self.dt = robot.Robot.loopTime
            _dt = self.dt
            _measurement = self.measurement
            self.state = subSystemState.Running
            robot.Robot.kalmanState = self.state
            self.UKF.predict(_dt)
            self.UKF.update(_measurement)
            # print("_measurement: " + str(_measurement))
            robot.Robot.xKalman = self.UKF.x[0]
            robot.Robot.yKalman = self.UKF.x[1]
            robot.Robot.robotAngle = np.degrees(self.UKF.x[4])
            print("Location Kalman: ( " + str(self.UKF.x[0]) + " , " + str(self.UKF.x[1]) + " ) m")
            print("Uncertainty Kalman: ( " + str(self.UKF.P[0][0]) + " , " + str(self.UKF.P[1][1]) + " ) m")
            # print("Velocity Kalman: ( " + str(self.UKF.x[2]) + " , " + str(self.UKF.x[3]) + " ) m/s")
            robot.Robot.uncertaintyX = self.UKF.P[0][0]
            robot.Robot.uncertaintyY = self.UKF.P[1][1]

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.kalmanState = self.state

    def updateModel(self, state, _dt):
        """
        This functions dives an estimate of the model at the next state.
        @param dt: time delta
        @return: state matrix at t=t+dt
        """

        _x = state
        # get angle for internal use
        _angle = state[4]

        # for startup
        if _dt > 1:
            _dt = 1

        # calculate velocity, used in several calculations
        # print("_Vx in state matrix: " + str(state[2]))
        _velocity = math.sqrt(math.pow(state[2], 2) + math.pow(state[3], 2))
        # print("_velocity: " + str(_velocity))

        # calculate the forces on the robot and add them together
        _fa = mathFunctions.motor_to_force(robot.Robot.input_motor)  # TODO: implement braking/reverse?
        # print("_fa: " + str(_fa))

        _fd = np.sign(_velocity) * (robot.Robot.b * np.abs(_velocity) + robot.Robot.c * np.power(_velocity, 2))
        # print("_fd: " + str(_fd))
        _fres = _fa - _fd

        # calculate acceleration, based on Newton's second law
        _a = _fres / robot.Robot.mass
        # print("_a: " + str(_a))
        # calculate new vx and vy
        _vX = (_velocity + _a * _dt) * math.sin(math.radians(_angle))
        _vY = (_velocity + _a * _dt) * math.cos(math.radians(_angle))

        # get steering angle
        _steeringAngle = mathFunctions.steer_to_angle(robot.Robot.input_servo, "degree")
        _r = 0
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

        # update to new state, only when t>0.2 for "reasons"
        _xNew = [[0], [0], [0], [0], [0]]
        # positions
        _xNew[0] = state[0] + _vX * _dt
        _xNew[1] = state[1] + _vY * _dt
        # velocity
        _xNew[2] = _vX
        self.vx_old = _vX
        # print("_vX: " + str(_vX))
        _xNew[3] = _vY
        # angle
        _xNew[4] = _angle
        return _xNew

    def state_mean(self, _sigmas, _Wm):
        """
        Used in kalman filter to calculate the state mean. Needed because angles can't be added properly
        @param _Wm:
        @return:
        """
        x = np.zeros(5)
        sum_sin, sum_cos = 0., 0.

        for i in range(len(_sigmas)):
            s = _sigmas[i]
            x[0] += s[0] * _Wm[i]
            x[1] += s[1] * _Wm[i]
            x[2] += s[2] * _Wm[i]
            x[3] += s[3] * _Wm[i]
            sum_sin += math.sin(math.radians(s[4])) * _Wm[i]
            sum_cos += math.cos(math.radians(s[4])) * _Wm[i]
        x[4] = math.atan2(sum_sin, sum_cos)
        return x

    def hx(self, x):
        return np.array([x[0], x[1]])
