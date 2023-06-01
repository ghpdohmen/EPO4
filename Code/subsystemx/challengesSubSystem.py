#inputs, starting position and destination
#outputs, vector with direction to be taken

import numpy as np
import robot
#import statemachine
from misc import mathFunctions
from subsystemx.subsystem import subSystem
from misc.robotModeEnum import robotMode
import purePursuit

mode = robot.Robot.operatingMode
match mode:
    case "ChallengeA":
        print("activating A")
        robot.Robot.operatingMode = robotMode.ChallengeA
        purePursuit.purePursuit.
        if mathFunctions.ish() and mathFunctions.ish():
            #stop
    case "ChallengeB":
        print("activating B")
        #start
        #drive
        #wait
        #drive
        #end
    case _:
        print("not supported")




