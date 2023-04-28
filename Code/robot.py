# holds all info of the robot and manages all subsystems
from misc.robotModeEnum import robotMode
from misc.robotStatusEnum import robotStatus
from subsystems.communication import communicationSubSystem
from subsystems.inputSubSystem import inputSubSystem
from subsystems.subsystemStateEnum import subSystemState
from subsystems.timing import timeSubSystem


class Robot:
    # current robot state
    operatingMode = robotMode.NotChosen
    status = robotStatus.Paused
    xCurrent = 0
    yCurrent = 0

    #audio stuff
    code = None #Expected in bytes!
    speakerOn = False
    carrierFrequency = 20000 # in Hz
    bitFrequency = 1000 # in Hz
    repetitionCount = 10 # in number of bits


    # subsystem states
    communicationState = subSystemState.Stopped
    timingState = subSystemState.Stopped
    inputState = subSystemState.Stopped

    # sensor values
    distanceLeft = 0
    distanceRight = 0
    batteryVoltage = 0

    # output values
    input_motor = 150;
    input_servo = 150;
    COMport = 'COM5'

    # timing
    runTime = 0  # time since hitting start (in seconds)
    loopTime = 0  # delta t, time in between updates

    def __init__(self, _xCurrent, _yCurrent):
        self.yCurrent = _yCurrent;
        self.xCurrent = _xCurrent;
        self.communicationSubSystem = communicationSubSystem()
        self.timeSubSystem = timeSubSystem()
        self.inputSubSystem = inputSubSystem()

    # start all subsystems
    def start(self):
        self.operatingMode = robotMode.Manual #TEMPORARY, REMOVE WHEN GUI IS IMPLEMENTED
        self.communicationSubSystem.start(self.COMport)
        self.timeSubSystem.start()
        self.inputSubSystem.start()

    # updates all subsystems
    def update(self):
        self.communicationSubSystem.update()
        self.timeSubSystem.update()
        self.inputSubSystem.update()
        print(self.batteryVoltage)

    def stop(self):
        self.communicationSubSystem.stop()
        self.timeSubSystem.stop()
        self.inputSubSystem.stop()
