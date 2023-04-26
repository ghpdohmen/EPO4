# holds all info of the robot and manages all subsystems
from subsystems.communication import communicationSubSystem


class Robot:
    xCurrent = 0
    yCurrent = 0
    input_motor = 135;
    input_servo = 135;

    def __init__(self, _xCurrent, _yCurrent):
        self.yCurrent = _yCurrent;
        self.xCurrent = _xCurrent;
        self.communicationSubSystem = communicationSubSystem()

    # start all subsystems
    def start(self):
        self.communicationSubSystem.start()

    def update(self):
        self.communicationSubSystem.update()

    def stop(self):
        self.communicationSubSystem.stop()
