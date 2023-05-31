import math

import numpy as np

import robot
from misc import mathFunctions
from subsystemx.subsystem import subSystem
from subsystemx.subsystemStateEnum import subSystemState


class modelSubSystem (subSystem):

    def __init__(self):
        self.fa = 0
        self.fb = 0
        self.w = 0
        self.vX = 0
        self.vY = 0
        self.steeringAngle = 0
        self.state = subSystemState.Started
        self.r = 0
        self.posx = 0
        self.posy = 0
        robot.Robot.modelState = self.state


    def update(self):
        self.state = subSystemState.Running
        #gathering values from robot
        dt = robot.Robot.loopTime/np.power(10,9)
        self.steeringAngle = mathFunctions.steer_to_angle(robot.Robot.input_servo, "degree")
        self.fa = mathFunctions.motor_to_force(robot.Robot.input_motor)


        self.fd = np.sign(robot.Robot.velocity) * (robot.Robot.b * np.abs(robot.Robot.velocity) + robot.Robot.c * np.power(robot.Robot.velocity, 2))
        self.fres = self.fa - self.fb - self.fd
        self.a = self.fres / robot.Robot.mass
        robot.Robot.velocity += self.a * dt
        if(np.abs(robot.Robot.velocity) < 0.005):
            robot.Robot.velocity = 0
        robot.Robot.robotAngle += 2*math.degrees(self.w) * dt
        self.posx = self.posx + self.vX * dt
        self.posy = self.posy + self.vY * dt
        self.vX = robot.Robot.velocity * math.cos(math.radians( robot.Robot.robotAngle))
        self.vY = robot.Robot.velocity * math.sin(math.radians( robot.Robot.robotAngle))
        #print("angle " + str(robot.Robot.robotAngle))
        #print(str(self.vX) + " , " + str(self.vY))
        #print(self.vX * dt)
        if self.steeringAngle != 0:
            self.r = robot.Robot.wheelBase / math.tan(math.radians(self.steeringAngle))  # goed
        #print(self.r)
        #print(self.steeringAngle)
        if (robot.Robot.velocity != 0) & (self.r != 0):
            self.w = robot.Robot.velocity / self.r
        else:
            self.w = 0
        robot.Robot.modelState = self.state
        #print("steeringAngle: " + str(self.steeringAngle))
        print("robotAngle: " + str(robot.Robot.robotAngle))
        print("vY: " + str(self.vY))
        print("locationMSS: (" + str(self.posx) + " , " + str(self.posy) + " )")
        #print("velocity: " + str(robot.Robot.velocity))
        #print("Fa: " + str(self.fa))
        robot.xCurrent = self.posx
        robot.yCurrent = self.posy #FIXME: this is somehow not updating on the robot?
        #print("Fb:" + str(self.fa))

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.modelState = self.state

