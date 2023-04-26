# prototype for all subsystems
from subsystems.subsystemStateEnum import subsystemState


class subSystem:
    state = subsystemState.Stopped

    # used for initializing the subsystem
    def __int__(self):
        self.state = subsystemState.Stopped

    # start the subsystem. For example, reset running averages here
    def start(self):
        self.state = subsystemState.Started

    # updates the subsystem, called in the main program loop
    def update(self):
        self.state = subsystemState.Running

    # stops the subsystem
    def stop(self):
        self.state = subsystemState.Stopped

    def getState(self):
        return self.state
