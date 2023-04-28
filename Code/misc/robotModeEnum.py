#enum used to keep track of the current program used by the robot, set by the GUI
from enum import Enum


class robotMode(Enum):
    Manual = 0
    ChallengeA = 1
    ChallengeB = 2
    ChallengeC = 3
    ChallengeD = 4
    ChallengeE = 5
    NotChosen = 6
    EStop = 7
