# prototype for all subsystems
from subsystems.subsystemStateEnum import subSystemState


class subSystem:
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
