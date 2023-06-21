# inputs, starting position and destination
# outputs, vector with direction to be taken

import robot
from misc import mathFunctions
from subsystemx.subsystem import subSystem
from misc.robotModeEnum import robotMode
from subsystemx.subsystemStateEnum import subSystemState


class challengesSubSystem(subSystem):
    mode = 0  # initialisation of some variables which have to be updated
    x_location = 0
    y_location = 0
    challenge_complete = False
    challenge_end = False

    def __init__(self):
        self.ch_startPos = []
        self.ch_aEnd = []
        self.ch_bMid = []
        self.ch_bEnd = []

        self.MOE = 30
        self.runtimeCheck = 0

        self.stateA = 0
        self.stateB = 0
        self.stateE = 0
        self.challengeELeft = False
        # self.challenge_end = False

    def start(self):
        if (robot.Robot.operatingMode == robotMode.Manual) | (robot.Robot.operatingMode == robotMode.EStop):
            # check to make sure the challenges are only excecuted when needed
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
            case robotMode.ChallengeA:
                #print("activating A")
                self.challengeA()
            case robotMode.ChallengeB:
                #print("activating B")
                self.challengeB()
            case robotMode.ChallengeE:
                self.challengeE()
            case _:
                pass

    def challengeA(self):
        # TODO: tune this
        """
        the state machine for Challenge A
        :return:
        """
        self.ch_startPos = robot.Robot.startPos
        self.ch_aEnd = robot.Robot.aEnd

        match self.stateA:
            case 0:
                # Initial state
                robot.Robot.startPos = self.ch_startPos
                robot.Robot.endPos = self.ch_aEnd
                print("Challenge A state 0")
                self.runtimeCheck = robot.Robot.runTime
                self.stateA = 1
                print("State a: " + str(self.stateA))

            case 1:
                # holding for 1 second to have the robot converge it's location estimate
                print("waiting, state 1")
                if (self.runtimeCheck + 3) <= robot.Robot.runTime:  # stands still for 10 seconds
                    self.stateA = 2

            case 2:
                # start moving forward with pure pursuit
                robot.Robot.input_motor = 160
                print("Challenge A state 2")
                self.stateA = 4

            case 3:
                # if we are within 1.5 meters of the goal, start slowing down
                if mathFunctions.ish(robot.Robot.posXLocalization * 100, self.ch_aEnd[0], 150) == True and mathFunctions.ish(
                        robot.Robot.posYLocalization * 100, self.ch_aEnd[1], 150) == True:
                    # self.challenge_complete = True
                    print("slowing down")
                    robot.Robot.input_motor = 150  # stops once at destination
                    self.stateA = 4
                else:
                    print("state 3")
                    pass
            case 4:
                # if we re within a meter of the goal, turn off the motor
                if mathFunctions.ish(robot.Robot.posXLocalization * 100, self.ch_aEnd[0], 75) == True and mathFunctions.ish(
                        robot.Robot.posYLocalization* 100, self.ch_aEnd[1], 75) == True:
                    # self.challenge_complete = True
                    print("slowing down")
                    robot.Robot.input_motor = 140  # stops once at desti
                    # nation
                    self.stateA = 5
                else:
                    print("state 4")
                    pass

            case 5:
                # if we are within 50 cm, reverse for 1 update cycle (approx 0.4 seconds)
                if mathFunctions.ish(robot.Robot.posXLocalization * 100, self.ch_aEnd[0], 50) == True and mathFunctions.ish(
                        robot.Robot.posYLocalization * 100, self.ch_aEnd[1], 50) == True:
                    print("slowing down")
                    robot.Robot.input_motor = 150  # stops once at destination
                    self.stateA = 6
                else:
                    print("state 5")
                    pass
                robot.Robot.input_motor = 150
            case 6:
                # Guess we are at our goal then.
                print("arrived at destination, woohoo")
                self.challenge_end = True
                robot.Robot.input_motor = 150  # stops once at destination

            case _:
                print("something went wrong in challenge A function")  # just in case

    def challengeB(self):
        #TODO: tune this
        """
        state machine for challenge B
        :return:
        """
        self.ch_startPos = robot.Robot.startPos
        self.ch_bMid = robot.Robot.bMid
        self.ch_bEnd = robot.Robot.bEnd

        match self.stateB:
            case 0:
                #Initial state
                robot.Robot.startPos = self.ch_startPos
                robot.Robot.endPos = self.ch_bMid

                self.runtimeCheck = robot.Robot.runTime
                self.stateB = 1

            case 1:
                print("waiting")
                #wait for a second to have the robot converge its location estimate
                if (self.runtimeCheck + 3) <= robot.Robot.runTime:  # stands still for 10 seconds
                    self.stateB = 2

            case 2:
                #start driving with pure pursuit enable
                robot.Robot.input_motor = 160
                self.stateB = 4

            case 3:
                #if we are withing 150cm of our waypount, slow down
                if mathFunctions.ish(robot.Robot.posXLocalization * 100, self.ch_bMid[0], 150) == True and mathFunctions.ish(
                        robot.Robot.posYLocalization * 100, self.ch_bMid[1], 150) == True:
                    robot.Robot.input_motor = 158
                    # self.runtimeCheck = robot.Robot.runTime
                    self.stateB = 4
                else:
                    pass

            case 4:
                #if we are within 100cm, turn off the motor
                if mathFunctions.ish(robot.Robot.posXLocalization * 100, self.ch_bMid[0], 100) == True and mathFunctions.ish(
                        robot.Robot.posYLocalization * 100, self.ch_bMid[1], 100) == True:
                    # print("arrived at waypoint, woohoo")
                    robot.Robot.input_motor = 150
                    # self.runtimeCheck = robot.Robot.runTime
                    self.stateB = 5
                else:
                    #  print("cry time")
                    pass

            case 5:
                #if we are withing 50 cm, brake for 1 update cycle.
                if mathFunctions.ish(robot.Robot.posXLocalization * 100, self.ch_bMid[0], 50) == True and mathFunctions.ish(
                        robot.Robot.posYLocalization * 100, self.ch_bMid[1], 50) == True:
                    # print("arrived at waypoint, woohoo")
                    robot.Robot.input_motor = 140
                    self.runtimeCheck = robot.Robot.runTime
                    self.stateB = 6
                else:
                    # print("cry time")
                    pass

            case 6:
                #we are probably there, let's stand still for 10 seconds
                robot.Robot.input_motor = 150
                print("position midpoint:" + str(robot.Robot.xCurrent) + " , " + str(robot.Robot.yCurrent))
                if (self.runtimeCheck + 10) <= robot.Robot.runTime:  # stands still for 10 seconds
                    self.stateB = 7

            case 7:
                #Resetting the path for the last bit
                robot.Robot.startPos = [robot.Robot.posXLocalization * 100, robot.Robot.posYLocalization * 100]  # sets new destination positions for second stint
                robot.Robot.endPos = self.ch_bEnd
                self.stateB = 8

            case 8:
                #let's start driving again
                robot.Robot.input_motor = 160
                self.stateB = 10

            case 9:
                #if we are within 150 cm slow down
                if mathFunctions.ish(robot.Robot.posXLocalization * 100, self.ch_bEnd[0], 150) == True and mathFunctions.ish(
                        robot.Robot.posYLocalization * 100, self.ch_bEnd[1], 150) == True:

                    robot.Robot.input_motor = 158  # stops once at destination
                    self.stateB = 10

                else:
                    #  print("cry time")
                    pass

            case 10:
                #if we are within 100 cm, turn off the motor
                if mathFunctions.ish(robot.Robot.posXLocalization * 100, self.ch_bEnd[0], 75) == True and mathFunctions.ish(
                        robot.Robot.posYLocalization * 100, self.ch_bEnd[1], 75) == True:

                    robot.Robot.input_motor = 150  # stops once at destination
                    self.stateB = 11

                else:
                    #  print("cry time")
                    pass

            case 11:
                #if we are within 50 cm, brake for 1 cycle
                if mathFunctions.ish(robot.Robot.posXLocalization * 100, self.ch_bEnd[0], 25) == True and mathFunctions.ish(
                        robot.Robot.posYLocalization * 100, self.ch_bEnd[1], 25) == True:

                    robot.Robot.input_motor = 140  # stops once at destination
                    self.stateB = 12

                else:
                    #  print("cry time")
                    pass

            case 12:
                #we should be at the waypoint now?
                print("arrived at destination, woohoo")
                self.challenge_end = True
                robot.Robot.input_motor = 150

            case _:
                print("something went wrong in challenge B function")  # just in case

    def challengeE(self):
        # TODO: finish this up.
        self.ch_startPos = robot.Robot.startPos
        self.ch_aEnd = robot.Robot.aEnd

        match self.stateE:
            case 0:
                robot.Robot.startPos = self.ch_startPos
                robot.Robot.endPos = self.ch_aEnd
                print("Challenge A state 0")
                self.runtimeCheck = robot.Robot.runTime
                self.stateE = 2

            case 1:
                print("waiting, state 1")
                if (self.runtimeCheck + 3) <= robot.Robot.runTime:  # stands still for 1 second
                    self.stateA = 2

            case 2:
                robot.Robot.input_motor = 160
                # Figure out if we are turning left or right
                if (self.ch_aEnd <= 240):
                    self.challengeELeft = False
                    robot.Robot.input_servo = 100
                else:
                    self.challengeELeft = True
                    robot.Robot.input_servo = 200

                print("Challenge E state 2")
                self.runtimeCheck = robot.Robot.runTime
                self.stateA = 3
            case 3:
                print("Challenge E state 3")
                if (self.runtimeCheck + 5) <= robot.Robot.runTime:  # after turning for 5 seconds, start pure pursuit
                    self.stateA = 4
                if self.challengeELeft:
                    robot.Robot.input_servo = 200
                else:
                    robot.Robot.input_servo = 100


            case 4:
                # start moving forward with pure pursuit
                robot.Robot.input_motor = 160
                print("Challenge E state 4")
                self.stateA = 4

            case 5:
                # if we are within 1.5 meters of the goal, start slowing down
                if mathFunctions.ish(self.x_location * 100, self.ch_aEnd[0], 100) == True and mathFunctions.ish(
                        self.y_location * 100, self.ch_aEnd[1], 100) == True:
                    # self.challenge_complete = True
                    print("slowing down")
                    robot.Robot.input_motor = 150  # stops once at destination
                    self.stateA = 5
                else:
                    print("state 5")
                    pass
            case 6:
                # if we re within a meter of the goal, turn off the motor
                if mathFunctions.ish(self.x_location * 100, self.ch_aEnd[0], 50) == True and mathFunctions.ish(
                        self.y_location * 100, self.ch_aEnd[1], 75) == True:
                    # self.challenge_complete = True
                    print("slowing down")
                    robot.Robot.input_motor = 140  # stops once at desti
                    # nation
                    self.stateA = 6
                else:
                    print("state6")
                    pass

            case 7:
                # if we are within 50 cm, reverse for 1 update cycle (approx 0.4 seconds)
                if mathFunctions.ish(self.x_location * 100, self.ch_aEnd[0], 50) == True and mathFunctions.ish(
                        self.y_location * 100, self.ch_aEnd[1], 50) == True:
                    print("slowing down")
                    robot.Robot.input_motor = 150  # stops once at destination
                    self.stateA = 8
                else:
                    print("state 7")
                    pass
                robot.Robot.input_motor = 150
            case 8:
                # Guess we are at our goal then.
                print("arrived at destination, woohoo")
                self.challenge_end = True
                robot.Robot.input_motor = 150  # stops once at destination

            case _:
                print("something went wrong in challenge A function")  # just in case

    def update(self):
        if self.state == subSystemState.Running or subSystemState.Started:
            self.state = subSystemState.Running

            self.mode = robot.Robot.operatingMode
            # print("RObot: " + str(robot.Robot.operatingMode))
            self.x_location = robot.Robot.xCurrent  # gets the KITT position from robot.py, constantly updated
            self.y_location = robot.Robot.yCurrent

            self.whichChallenge()  # constantly update to check whether a challenge ahs been selected
            # print("Challenge subsystem started")

        robot.Robot.challengesState = self.state

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.challengesState = self.state
