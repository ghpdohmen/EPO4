# holds all info of the robot and manages all subsystemx
from misc.robotModeEnum import robotMode
from misc.robotStatusEnum import robotStatus
from subsystemx.communication import communicationSubSystem
from subsystemx.csvLoggingSubSystem import csvLoggingSubsystem
from subsystemx.inputSubSystem import inputSubSystem
from subsystemx.localizationsubsystem import LocalizationSubSystem
from subsystemx.subsystemStateEnum import subSystemState
from subsystemx.timing import timeSubSystem


class Robot:
    # current robot state
    operatingMode = robotMode.Manual
    status = robotStatus.Paused
    xCurrent = 0
    yCurrent = 0

    # audio stuff
    code = "EB3A994F"  # String, hexadecimal
    speakerOn = False
    carrierFrequency = 6000  # in Hz
    bitFrequency = 2000  # in Hz
    repetitionCount = 64  # in number of bits

    # subsystem states
    communicationState = subSystemState.Stopped
    timingState = subSystemState.Stopped
    inputState = subSystemState.Stopped
    localizationState = subSystemState.Stopped
    loggingState = subSystemState.Stopped

    # sensor values
    distanceLeft = 0
    distanceRight = 0
    batteryVoltage = 0

    # output values
    input_motor = 150;
    input_servo = 150;
    COMport = 'COM6'  # TODO: enable in GUI

    # timing
    runTime = 0  # time since hitting start (in seconds)
    loopTime = 0  # delta t, time in between updates
    averageLoop = 0.1
    index = 0

    def __init__(self, _xCurrent, _yCurrent):
        self.yCurrent = _yCurrent;
        self.xCurrent = _xCurrent;
        self.communicationSubSystem = communicationSubSystem()
        self.timeSubSystem = timeSubSystem()
        self.inputSubSystem = inputSubSystem()
        self.loggingSubSystem = csvLoggingSubsystem()
        self.localizationSubSystem = LocalizationSubSystem()

    # start all subsystemx
    def start(self, _operatingMode):
        self.operatingMode = _operatingMode
        # let's quickly check if we have set an operating mode, otherwise running the robot is so hard
        if _operatingMode == robotMode.NotChosen:
            # TODO: implement error message in gui
            print("no operating mode chosen")
            return

        #self.localizationSubSystem.start()
        self.communicationSubSystem.start(self.COMport)
        self.timeSubSystem.start()
        self.inputSubSystem.start()
        self.loggingSubSystem.start()


        # printing the loop time, so we can optimize this via multithreading
        print(self.loopTime)
        self.status = robotStatus.Running

    # updates all subsystemx
    def update(self):
        if (self.status == robotStatus.Running) | (self.status == robotStatus.Paused) | (
                self.status == robotStatus.Planning):
            #print("update")
            self.timeSubSystem.update()
            #print("time")
            self.inputSubSystem.update()
            #print("input")
            self.loggingSubSystem.update()
            #print("logging")
            self.communicationSubSystem.update()
            #("comms")
            self.localizationSubSystem.update()
            #print(self.distanceLeft)
            # printing the loop time, so we can optimize this via multithreading

            if self.loopTime != 0:
                if self.index == 0:
                    self.index = 1
                    return
                self.index += 1
                self.averageLoop = self.averageLoop + (self.loopTime / 1000000000 - self.averageLoop) / self.index
                #print("average loop time:" + str(self.averageLoop) + " s")
                print("update frequency" + str(1/self.averageLoop) + " Hz ")

    def stop(self):
        self.communicationSubSystem.stop()
        self.timeSubSystem.stop()
        self.inputSubSystem.stop()
        self.loggingSubSystem.stop()
