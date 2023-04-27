#enum used to keep track of the current program used by the robot, set by the GUI
from enum import Enum


class robotMode(Enum):
    Manual = 1
    ChallengeA = 2
    ChallengeB = 3
    ChallengeC = 4
    ChallengeD = 5
    ChallengeE = 6
    NotChosen = 7
    EStop = 8
