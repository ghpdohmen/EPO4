import csv
import time

import robot
from subsystemx.subsystem import subSystem
from subsystemx.subsystemStateEnum import subSystemState


# TODO: comments toevoegen
class csvLoggingSubsystem(subSystem):
    """
    This subsystem is used for constantly writing the current state to a .csv file for debugging purposes
    """

    def __init__(self):
        self.state = subSystemState.Stopped
        self.file = open(time.strftime(str(time.time())) + '.csv',
                         'w')  # create a file with the current time as name, so we can find the correct one later.

    def start(self):
        self.state = subSystemState.Started
        robot.Robot.loggingState = self.state
        self.writer = csv.writer(self.file)  # create a single writer, this is used each update
        self.writer.writerow(['Time', 'Voltage', 'SensorLeft', 'SensorRight', 'MotorPower',
                              'ServoPower'])  # write the top row, used for identification

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
