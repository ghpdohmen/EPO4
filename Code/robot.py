# holds all info of the robot and manages all subsystems
from subsystems.communication import communicationSubSystem
from subsystems.subsystemStateEnum import subSystemState


class Robot:
    # current robot state
    xCurrent = 0
    yCurrent = 0

    # subsystem states
    communicationState = subSystemState.Stopped
    timingState = subSystemState.Stopped

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
        self.communicationSubSystem = communicationSubSystem(self.COMport)

    # start all subsystems
    def start(self):
        self.communicationSubSystem.start()

    # updates all subsystems
    def update(self):
        self.communicationSubSystem.update()

    def stop(self):
        self.communicationSubSystem.stop()
