#enum used in subsystem class to keep track of subsystem state
from enum import Enum

class subsystemState(Enum):
    Stopped = 1
    Started = 2
    Running = 3
    Crashed = 4
    ReadyForUpdate = 5
    WaitingForData = 6

