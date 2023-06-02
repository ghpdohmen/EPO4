import csv
import time

import robot
from subsystemx.subsystem import subSystem
from subsystemx.subsystemStateEnum import subSystemState


# TODO: comments toevoegen
class csvLoggingSubsystem(subSystem):

    def __init__(self):
        self.state = subSystemState.Stopped
        self.file = open(time.strftime(str(time.time())) + '.csv', 'w')

    def start(self):
        self.state = subSystemState.Started
        robot.Robot.loggingState = self.state
        self.writer = csv.writer(self.file)
        self.writer.writerow(['Time', 'Voltage', 'SensorLeft', 'SensorRight', 'MotorPower', 'ServoPower'])

    def update(self):
        if (self.state == subSystemState.Started) | (self.state == subSystemState.Running):
            self.state = subSystemState.Running
            robot.Robot.loggingState = self.state
            self.writer.writerow(
                [str(robot.Robot.runTime), str(robot.Robot.batteryVoltage), str(robot.Robot.distanceLeft),
                 str(robot.Robot.distanceRight), str(robot.Robot.input_motor), str(robot.Robot.input_servo)])

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.loggingState = self.state
        print("Closing CSV")
        self.file.close()
