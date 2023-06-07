# inputs, starting position and destination
# outputs, vector with direction to be taken

import robot
from misc import mathFunctions
from subsystemx.subsystem import subSystem
from misc.robotModeEnum import robotMode
from subsystemx.subsystemStateEnum import subSystemState


class challengesSubSystem(subSystem):
    mode = 0            # initialisation of some variables which have to be updated
    x_location = 0
    y_location = 0

    def __init__(self):
        self.ch_startPos = []
        self.ch_aEnd = []
        self.ch_bMid = []
        self.ch_bEnd = []

        self.MOE = 0
        self.runtimeCheck = 0

        self.stateA = int
        self.stateB = int

    def start(self):
        if (robot.Robot.operatingMode == robotMode.Manual) | (robot.Robot.operatingMode == robotMode.EStop):  # check to make sure program only runs when needed
            self.state = subSystemState.Stopped
        else:
            self.state = subSystemState.Started

        robot.Robot.challengesState = self.state

    def whichChallenge(self):
        """
        chooses which challenge has to be run depending on the operating mode
        :return:
        """
        match self.mode:
            case "ChallengeA":
                print("activating A")
                self.challengeA()
            case "ChallengeB":
                print("activating B")
                self.challengeB()
            case _:
                pass

    def challengeA(self):
        """
        the state machine for Challenge A
        :return:
        """
        self.ch_startPos = robot.Robot.startPos
        self.ch_aEnd = robot.Robot.aEnd

        self.stateA = 0

        match self.stateA:
            case 0:
                robot.Robot.startPos = self.ch_startPos
                robot.Robot.endPos = self.ch_aEnd
                self.stateA = 1

            case 1:
                robot.Robot.input_motor = 160
                self.stateA = 2

            case 2:
                if mathFunctions.ish(self.x_location, self.ch_aEnd[0], self.MOE) == True and mathFunctions.ish(self.y_location, self.ch_aEnd[1], self.MOE) == True:
                    print("arrived at destination, woohoo")
                    robot.Robot.input_motor = 150  # stops once at destination
                else:
                    #  print("cry time")
                    pass

            case _:
                print("something went wrong in challenge A function")  # just in case

    def challengeB(self):
        """
        state machine for challenge B
        :return:
        """
        self.ch_startPos = robot.Robot.startPos
        self.ch_bMid = robot.Robot.bMid
        self.ch_bEnd = robot.Robot.bEnd

        self.stateB = 0

        match self.stateB:
            case 0:
                robot.Robot.startPos = self.ch_startPos
                robot.Robot.endPos = self.ch_bMid
                self.stateB = 1

            case 1:
                robot.Robot.input_motor = 160
                self.stateB = 2

            case 2:
                if mathFunctions.ish(self.x_location, self.ch_bMid[0], self.MOE) == True and mathFunctions.ish(self.y_location, self.ch_bMid[1], self.MOE) == True:
                    print("arrived at waypoint, woohoo")
                    robot.Robot.input_motor = 150
                    self.runtimeCheck = robot.Robot.runTime
                    self.stateB = 3
                else:
                    #  print("cry time")
                    pass

            case 3:
                if (self.runtimeCheck + 10) <= robot.Robot.runTime:  # stands still for 10 seconds
                    self.stateB = 4

            case 4:
                robot.Robot.startPos = self.ch_bMid  # sets new destination positions for second stint
                robot.Robot.endPos = self.ch_bEnd
                self.stateB = 5

            case 5:
                robot.Robot.input_motor = 160
                self.stateB = 6

            case 6:
                if mathFunctions.ish(self.x_location, self.ch_bEnd[0], self.MOE) == True and mathFunctions.ish(self.y_location, self.ch_bEnd[1], self.MOE) == True:
                    print("arrived at destination, woohoo")
                    robot.Robot.input_motor = 150  # stops once at destination
                else:
                    #  print("cry time")
                    pass

            case _:
                print("something went wrong in challenge B function")  # just in case

    def update(self):
        if self.state == subSystemState.Running or subSystemState.Started:
            self.state = subSystemState.Running

            self.mode = robot.Robot.operatingMode

            self.x_location = 0  # gets the KITT position from robot.py, constantly updated
            self.y_location = 0

            self.whichChallenge()  # constantly update to check whether a challenge ahs been selected

        robot.Robot.challengesState = self.state

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.challengesState = self.state
