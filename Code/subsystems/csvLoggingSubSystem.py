import csv
import time

import robot
from subsystems.subsystem import subSystem
from subsystems.subsystemStateEnum import subSystemState


class csvLoggingSubsystem(subSystem):

    def __init__(self):
        self.state = subSystemState.Stopped
        self.file = open('C:/Users/guusd/Desktop/log' + time.strftime(str(time.time()))+'.csv', 'w')

    def start(self):
        self.state = subSystemState.Started
        robot.Robot.loggingState = self.state
        self.writer = csv.writer(self.file)
        self.writer.writerow(['Time','Voltage','SensorLeft','SensorRight', 'MotorPower', 'ServoPower'])

    def update(self):
        self.state = subSystemState.Running
        robot.Robot.loggingState = self.state
        self.writer.writerow([str(robot.Robot.runTime), str(robot.Robot.batteryVoltage),str(robot.Robot.distanceLeft), str(robot.Robot.distanceRight), str(robot.Robot.input_motor), str(robot.Robot.input_servo)])

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.loggingState = self.state
        print("CLosing CSV")
        self.file.close()


