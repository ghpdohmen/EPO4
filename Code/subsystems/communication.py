# communication subsystem. Handles all communication with KITT
import serial as serial

import robot
from subsystems.subsystem import subSystem


class communicationSubSystem(subSystem):
    # variable declaration
    serial_port = 0
    comport = 'COM3'
    baud_rate = 115200

    # starts communication with the robot
    def start(self):
        self.serial_port = serial.Serial(self.comport, self.baud_rate, rtscts=True)

    # sends all new data to the robot
    def update(self):
        self.serial_port.write(b'M' + robot.input_motor + '\n')
        self.serial_port.write(b'D' + robot.input_servo + '\n')

    #stop all communication
    def stop(self):
        self.serial_port.close()
