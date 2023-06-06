# enum used in subsystem class to keep track of subsystem state
from enum import Enum


class subSystemState(Enum):
    """
    This enum is used for keeping track of the different states of a subsystem.
    These are mainly used for debugging
    """
    Stopped = 1
    Started = 2
    Running = 3
    Crashed = 4
    ReadyForUpdate = 5
    WaitingForData = 6
