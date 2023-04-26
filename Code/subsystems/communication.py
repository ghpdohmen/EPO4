# communication subsystem. Handles all communication with KITT
import serial as serial

import robot
from subsystems.subsystem import subSystem
from subsystems.subsystemStateEnum import subSystemState


# communication subsystem. Handles all communication with KITT
class communicationSubSystem(subSystem):
    # variable declaration
    serial_port = 0
    baud_rate = 115200

    # constructor, sets the comport defined in robot
    def __int__(self, _comPort):
        self.comport = _comPort

    # starts communication with the robot
    def start(self):
        self.state = subSystemState.Started
        self.serial_port = serial.Serial(self.comport, self.baud_rate, rtscts=True)
        self.state = subSystemState.ReadyForUpdate

    # sends all new data to the robot en gets new data from the robot
    def update(self):

        # for debugging: print the current state of the comms subsystem
        print('Comms state:' + str(self.state))
        # set state in robot, so this can be read in gui
        robot.Robot.communicationState = self.state

        # to make sure that we don't interrupt reading the data, some extra states have been added
        if self.state == subSystemState.ReadyForUpdate:
            self.state = subSystemState.Running
            # writing the current pwm signal for both the motor and steering
            self.serial_port.write(b'M' + bytes(str(robot.Robot.input_motor), 'ascii') + b'\n')
            self.serial_port.write(b'D' + bytes(str(robot.Robot.input_servo), 'ascii') + b'\n')

            # start sending the data and set our state to WaitingForData, which we will stay in until the packet is
            # complete
            self.serial_port.write(b'S\n')
            self.state = subSystemState.WaitingForData

        # Here we receive the data and split it in the respective variables for the robot class
        if self.state == subSystemState.WaitingForData:
            _incomingData = self.serial_port.read_until(b'\x04')
            _incomingDataSplit = str(_incomingData).split('\\n')

            #distance sensors
            _sensors = _incomingDataSplit[12]
            _distance = _sensors.split(' ')
            robot.Robot.distanceLeft = _distance[2]
            robot.Robot.distanceRight = _distance[4]
            print(_sensors)

            # set our state back to readyforupdate, so we can again send and receive data
            self.state = subSystemState.ReadyForUpdate

    # stop all communication
    def stop(self):
        self.state = subSystemState.Stopped
        self.serial_port.close()
