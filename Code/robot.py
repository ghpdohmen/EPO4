# holds all info of the robot and manages all subsystems
from misc.robotModeEnum import robotMode
from misc.robotStatusEnum import robotStatus
from subsystems.communication import communicationSubSystem
from subsystems.csvLoggingSubSystem import csvLoggingSubsystem
from subsystems.inputSubSystem import inputSubSystem
from subsystems.subsystemStateEnum import subSystemState
from subsystems.timing import timeSubSystem


class Robot:
    # current robot state
    operatingMode = robotMode.Manual
    status = robotStatus.Paused
    xCurrent = 0
    yCurrent = 0

    # audio stuff
    code = "A23"  # String, hexadecimal
    speakerOn = False
    carrierFrequency = 10000  # in Hz
    bitFrequency = 1000  # in Hz
    repetitionCount = 32  # in number of bits

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
    COMport = 'COM5'  # TODO: enable in GUI

    # timing
    runTime = 0  # time since hitting start (in seconds)
    loopTime = 0  # delta t, time in between updates

    def __init__(self, _xCurrent, _yCurrent):
        self.yCurrent = _yCurrent;
        self.xCurrent = _xCurrent;
        self.communicationSubSystem = communicationSubSystem()
        self.timeSubSystem = timeSubSystem()
        self.inputSubSystem = inputSubSystem()
        self.loggingSubSystem = csvLoggingSubsystem()
        # self.localizationSubSystem = LocalizationSubSystem()

    # start all subsystems
    def start(self, _operatingMode):
        self.operatingMode = _operatingMode
        # let's quickly check if we have set an operating mode, otherwise running the robot is so hard
        if _operatingMode == robotMode.NotChosen:
            # TODO: implement error message in gui
            print("no operating mode chosen")
            return

        self.communicationSubSystem.start(self.COMport)
        self.timeSubSystem.start()
        self.inputSubSystem.start()
        self.loggingSubSystem.start()
        # self.localizationSubSystem.start()

        # printing the loop time, so we can optimize this via multithreading
        print(self.loopTime)

    # updates all subsystems
    def update(self):
        if (self.status == robotStatus.Running) | (self.status == robotStatus.Paused) | (
                self.status == robotStatus.Planning):
            self.timeSubSystem.update()
            self.inputSubSystem.update()
            self.loggingSubSystem.update()
            # self.localizationSubSystem.update()
            # print(self.distanceLeft)
            # printing the loop time, so we can optimize this via multithreading
            if self.loopTime != 0:
                print(str(self.loopTime / 1000000) + " ms")

    def stop(self):
        self.communicationSubSystem.stop()
        self.timeSubSystem.stop()
        self.inputSubSystem.stop()
        self.loggingSubSystem.stop()
