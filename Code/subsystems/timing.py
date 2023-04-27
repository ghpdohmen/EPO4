import time

import robot
from subsystems.subsystem import subSystem
from subsystems.subsystemStateEnum import subSystemState


# responsible for keeping track of the runtime of the program and giving a delta time
class timeSubSystem(subSystem):
    startTime = 0
    previousTime = 0

    def start(self):
        self.state = subSystemState.Started
        self.startTime = time.time_ns()
        robot.Robot.timingState = self.state

    def update(self):
        robot.Robot.timingState = self.state
        if (self.startTime == 0) | (self.state == subSystemState.Crashed):
            self.state = subSystemState.Crashed
        else:
            _currTime = time.time_ns()  # caching variable, time.times_ns call might be a bit slower
            self.state = subSystemState.Running

            robot.Robot.runTime = (_currTime - self.startTime) / 1000000000  # dividing by 10^9, since
            # runTime is in seconds
            robot.Robot.loopTime = _currTime - self.previousTime  # delta t
            self.previousTime = _currTime  # set previous time to current time, so we can calculate looptime again
