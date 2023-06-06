# prototype for all subsystemx
from subsystemx.subsystemStateEnum import subSystemState


class subSystem:
    """
    The subsystem class is the prototype class used for all subsystems. This gives a clear and cohesive system for the robot code
    """
    state = subSystemState.Stopped

    # used for initializing the subsystem
    def __int__(self):
        self.state = subSystemState.Stopped

    # start the subsystem. For example, reset running averages here
    def start(self):
        self.state = subSystemState.Started

    # updates the subsystem, called in the main program loop
    def update(self):
        self.state = subSystemState.Running

    # stops the subsystem
    def stop(self):
        self.state = subSystemState.Stopped

    def getState(self):
        return self.state
