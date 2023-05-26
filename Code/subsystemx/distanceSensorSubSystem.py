from collections import deque

import numpy as np
from numpy import mean

import robot
from subsystemx.subsystem import subSystem
from subsystemx.subsystemStateEnum import subSystemState


class distanceSensorSubSystem(subSystem):
    """"
    This class is responsible for keeping track of a moving mean of the distance sensor measurements.
    This is done to filter out noisy measurements """
    distanceLeft = 0
    distanceRight = 0

    def __init__(self):
        self.distanceLeft = deque()  # using a linked list
        self.distanceRight = deque()

    def start(self):
        self.state = subSystemState.Started
        robot.Robot.distanceSensorState = self.state

    def update(self):
        self.state = subSystemState.Running
        # add new measurements to the front of the queue
        self.distanceLeft.appendleft(robot.Robot.distanceLeftRaw)
        self.distanceRight.appendleft(robot.Robot.distanceRightRaw)

        # if the length is greater than 5, then remove measurements from the end
        if self.distanceLeft.count() > 5:
            self.distanceLeft.pop()
        if self.distanceRight.count() > 5:
            self.distanceRight.pop()

        robot.Robot.distanceLeft = mean(self.distanceLeft)
        robot.Robot.distanceRight = mean(self.distanceRight)
        robot.Robot.distanceSensorState = self.state

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.distanceSensorState = self.state
