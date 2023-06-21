# communication subsystem. Handles all communication with KITT
import serial as serial

import robot
from subsystemx.subsystem import subSystem
from subsystemx.subsystemStateEnum import subSystemState
from subsystemx.challengesSubSystem import challengesSubSystem


# communication subsystem. Handles all communication with KITT
class communicationSubSystem(subSystem):
    # variable declaration
    serial_port = 0
    baud_rate = 115200
    comport = None

    def __int__(self):
        self.state = subSystemState.Stopped

    # starts communication with the robot
    def start(self, _comport):
        """
        Used for starting communication with the robot
        @param _comport: THe COMport used to communicate with the robot. Please give the OUTGOING COMport as given by your bluetooth settings
        """
        #TODO: Check if comport actually exists
        self.comport = _comport
        self.state = subSystemState.Started
        self.serial_port = serial.Serial(self.comport, self.baud_rate, rtscts=True) #TODO: error handling and inform user via GUI
        self.state = subSystemState.ReadyForUpdate
        print("started comms")

        #added from here:
        _carrier = (6000).to_bytes(2, byteorder='big')
        self.serial_port.write(b'F' + _carrier + b'\n')

        _bitFrequency = (2000).to_bytes(2, byteorder='big')
        self.serial_port.write(b'B' + _bitFrequency + b'\n')

        _repetition = (300).to_bytes(2, byteorder='big')
        # _repetition = (31.25).to_bytes(2, byteorder='big')
        self.serial_port.write(b'R' + _repetition + b'\n')

        code = (0xEB3A994F).to_bytes(4, byteorder='big')
        self.serial_port.write(b'C' + code + b'\n')

    # sends all new data to the robot en gets new data from the robot
    def update(self):
        # for debugging: print the current state of the comms subsystem
        # print('Comms state:' + str(self.state))

        # set state in robot, so this can be read in gui
        robot.Robot.communicationState = self.state

        # if challengesSubSystem.challenge_complete == False:
        # to make sure that we don't interrupt reading the data, some extra states have been added
        if self.state == subSystemState.ReadyForUpdate:
            self.state = subSystemState.Running

            if robot.Robot.instance.challengesSubSystem.challenge_end:
                # print("het werkt wel")
                # added from here:
                _carrier = (6000).to_bytes(2, byteorder='big')
                self.serial_port.write(b'F' + _carrier + b'\n')

                _bitFrequency = (2000).to_bytes(2, byteorder='big')
                self.serial_port.write(b'B' + _bitFrequency + b'\n')

                _repetition = (32).to_bytes(2, byteorder='big')
                # _repetition = (31.25).to_bytes(2, byteorder='big')
                self.serial_port.write(b'R' + _repetition + b'\n')

                code = (0xEB3A994F).to_bytes(4, byteorder='big')
                self.serial_port.write(b'C' + code + b'\n')
                # writes all audio commands
                # try writing everything at once
                if robot.Robot.speakerOn:
                    #print("writing speaker on")
                    self.serial_port.write(
                        b'A1\n' + b'M' + bytes(str(robot.Robot.input_motor), 'ascii') + b'\n' + b'D' + bytes(
                            str(robot.Robot.input_servo), 'ascii') + b'\n' + b'S\n')  # +
                else:
                    # print("writing speaker off")
                    self.serial_port.write(
                        b'A0\n' + b'M' + bytes(str(robot.Robot.input_motor), 'ascii') + b'\n' + b'D' + bytes(
                            str(robot.Robot.input_servo), 'ascii') + b'\n' + b'S\n')


            else:
                #print("het werkt niet")
                # writes all audio commands
                # try writing everything at once
                if robot.Robot.speakerOn:
                    # print("writing speaker on")
                    self.serial_port.write(
                        b'A1\n' + b'M' + bytes(str(robot.Robot.input_motor), 'ascii') + b'\n' + b'D' + bytes(
                            str(robot.Robot.input_servo), 'ascii') + b'\n' + b'S\n')  # +
                else:
                    # print("writing speaker off")
                    self.serial_port.write(
                        b'A0\n' + b'M' + bytes(str(robot.Robot.input_motor), 'ascii') + b'\n' + b'D' + bytes(
                            str(robot.Robot.input_servo), 'ascii') + b'\n' + b'S\n')


            # start sending the data and set our state to WaitingForData, which we will stay in until the packet is
            # complete
            #self.serial_port.write(b'S\n')
            self.state = subSystemState.WaitingForData
            #print("finished writing")




        # Here we receive the data and split it in the respective variables for the robot class
        if self.state == subSystemState.WaitingForData:
            #print("waiting for read")
            _incomingData = self.serial_port.read_until(b'\x04')
            _incomingDataSplit = str(_incomingData).split('\\n')

            # debug: print _incomingDataSplit to figure out which strings to parse.
            #print(_incomingData)

            # distance sensors
            _sensors = _incomingDataSplit[12]
            _distance = _sensors.split(' ')
            robot.Robot.distanceLeftRaw = int(_distance[3])
            robot.Robot.distanceRightRaw = int(_distance[5])

            # voltage sensor
            _voltage = _incomingDataSplit[13]
            _voltageSplit = _voltage.split(' ')
            robot.Robot.batteryVoltage = float(_voltageSplit[2])

            # set our state back to readyforupdate, so we can again send and receive data
            self.state = subSystemState.ReadyForUpdate

    # stop all communication
    def stop(self):
        self.state = subSystemState.Stopped
        self.serial_port.close()
