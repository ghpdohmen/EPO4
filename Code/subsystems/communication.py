# communication subsystem. Handles all communication with KITT
import serial as serial

import robot
from subsystems.subsystem import subSystem
from subsystems.subsystemStateEnum import subsystemState


class communicationSubSystem(subSystem):
    # variable declaration
    serial_port = 0
    comport = 'COM5'
    baud_rate = 115200

    # starts communication with the robot
    def start(self):
        self.state = subsystemState.Started
        self.serial_port = serial.Serial(self.comport, self.baud_rate, rtscts=True)
        self.state = subsystemState.ReadyForUpdate

    # sends all new data to the robot en gets new data from the robot
    def update(self):
        print('Comms state:' + str(self.state))
        if self.state == subsystemState.ReadyForUpdate:
            self.state = subsystemState.Running
            self.serial_port.write(b'M'+ bytes(str(robot.Robot.input_motor),'ascii') + b'\n')
            print(str(robot.Robot.input_motor))
            print(b'M'+ robot.Robot.input_motor.to_bytes(7, byteorder='big') + b'\n')
            self.serial_port.write(b'D'+ robot.Robot.input_servo.to_bytes(7, byteorder='big')+  b'\n')
            self.serial_port.write(b'S\n')
            self.state = subsystemState.WaitingForData
        if self.state == subsystemState.WaitingForData:
            _incomingData = self.serial_port.read_until(b'\x04')
            print(_incomingData)
            self.state = subsystemState.ReadyForUpdate


    #stop all communication
    def stop(self):
        self.state = subsystemState.Stopped
        self.serial_port.close()
