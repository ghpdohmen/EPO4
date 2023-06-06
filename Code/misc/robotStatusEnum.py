from enum import Enum


class robotStatus(Enum):
    """
    Used to give the current status of the robot
    """
    Planning = 1
    Running = 2
    Paused = 3
    Completed = 4
    ESTOP = 5
