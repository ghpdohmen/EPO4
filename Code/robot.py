# holds all info of the robot and manages all subsystemx
from misc.robotModeEnum import robotMode
from misc.robotStatusEnum import robotStatus
from subsystemx.communication import communicationSubSystem
from subsystemx.csvLoggingSubSystem import csvLoggingSubsystem
from subsystemx.distanceSensorSubSystem import distanceSensorSubSystem
from subsystemx.inputSubSystem import inputSubSystem
from subsystemx.kalman import kalman
from subsystemx.localizationsubsystem import LocalizationSubSystem
from subsystemx.modelSubSystem import modelSubSystem
from subsystemx.purePursuit import purePursuit
from subsystemx.challengesSubSystem import challengesSubSystem
from subsystemx.subsystemStateEnum import subSystemState
from subsystemx.timing import timeSubSystem


class Robot:
    # current robot state
    operatingMode = robotMode.Manual
    status = robotStatus.Paused
    xCurrent = 0 #in meters, updated from kalman filter
    yCurrent = 0 #in meters, updated from kalman filter
    uncertaintyX = 0 # in meters, updated from kalman filter
    uncertaintyY = 0 # in meters, updated from kalman filter
    robotAngle = 0 # in degrees
    velocity = 0 #TODO: delete this when modelsubsystem is deleted.
    speakerOn = False

    # challenge locations
    startPos = []
    endPos = []
    aEnd = []
    bMid = []
    bEnd = []


    # subsystem states
    communicationState = subSystemState.Stopped
    timingState = subSystemState.Stopped
    inputState = subSystemState.Stopped
    localizationState = subSystemState.Stopped
    loggingState = subSystemState.Stopped
    distanceSensorState = subSystemState.Stopped
    modelState = subSystemState.Stopped
    purePursuitState = subSystemState.Stopped
    kalmanState = subSystemState.Stopped
    challengesState = subSystemState.Stopped


    # sensor values
    distanceLeft = 0 #averaged over the last 5 cycles by distanceSensorSubSystem
    distanceLeftRaw = 0
    distanceRight = 0 #averaged over the last 5 cycles by distanceSensorSubSystem
    distanceRightRaw = 0
    batteryVoltage = 0
    posXLocalization = 0 # in meters
    posYLocalization = 0 # in meters

    # output values
    input_motor = 150
    input_servo = 150
    COMport = 'COM5'

    # timing
    runTime = 0  # time since hitting start (in seconds)
    loopTime = 0  # delta t, time in between updates
    averageLoop = 0.1
    index = 0

    #robot constants
    wheelBase = 0.335 #in meters
    mass = 5.6 #in kg
    faMax = 21 # in N #TODO: implement fa for different velocities
    fbMax = -21 # in N #TODO: tune me!
    b = 15 # Nm/s Viscous friction coefficient
    c = 0.08 # Nm/s Air drag coefficient


    def __init__(self, _xCurrent, _yCurrent):
        self.yCurrent = _yCurrent
        self.xCurrent = _xCurrent
        self.communicationSubSystem = communicationSubSystem()
        self.timeSubSystem = timeSubSystem()
        self.inputSubSystem = inputSubSystem()
        self.loggingSubSystem = csvLoggingSubsystem()
        self.localizationSubSystem = LocalizationSubSystem()
        #self.modelSubSystem = modelSubSystem()
        self.distanceSensorSubSystem = distanceSensorSubSystem()
        self.purePursuitSubSystem = purePursuit()
        self.challengesSubSystem = challengesSubSystem()
        self.kalmanSubSystem = kalman()

    # start all subsystemx
    def start(self, _operatingMode):
        self.operatingMode = _operatingMode
        # let's quickly check if we have set an operating mode, otherwise running the robot is so hard
        if _operatingMode == robotMode.NotChosen:
            # TODO: implement error message in gui
            print("no operating mode chosen")
            return


        #self.localizationSubSystem.start()
        self.communicationSubSystem.start(self.COMport)
        self.timeSubSystem.start()
        self.inputSubSystem.start()
        self.loggingSubSystem.start()
        self.distanceSensorSubSystem.start()
        self.purePursuitSubSystem.start()
        self.challengesSubSystem.start()
       # self.localizationSubSystem.start()
        self.kalmanSubSystem.start()
        # printing the loop time, so we can optimize this via multithreading
        #print(self.loopTime)
        self.status = robotStatus.Running

    # updates all subsystemx
    def update(self):
        if (self.status == robotStatus.Running) | (self.status == robotStatus.Paused) | (
                self.status == robotStatus.Planning):
            #print("update")
            self.timeSubSystem.update()
            self.inputSubSystem.update()
            #self.modelSubSystem.update()
            print("Runtime: " + str(self.runTime))
            #self.localizationSubSystem.update()
            #self.distanceSensorSubSystem.update()
            self.kalmanSubSystem.update()
            self.challengesSubSystem.update()
            self.purePursuitSubSystem.update()
            self.communicationSubSystem.update()
            self.loggingSubSystem.update()

            # printing the loop time, so we can optimize this via multithreading

            if self.loopTime != 0:
                if self.index == 0:
                    self.index = 1
                    return
                self.index += 1
                self.averageLoop = self.averageLoop + (self.loopTime - self.averageLoop) / self.index
                #print("average loop time:" + str(self.averageLoop) + " s")
                print("update frequency" + str(1/self.averageLoop) + " Hz ")

            #printing the robot location:
            print("Location: ( " + str(self.xCurrent) + " , " + str(self.yCurrent) + " )" )



    def stop(self):
        self.communicationSubSystem.stop()
        self.timeSubSystem.stop()
        self.inputSubSystem.stop()
        self.loggingSubSystem.stop()
        #self.modelSubSystem.stop()
        self.distanceSensorSubSystem.stop()
        self.purePursuitSubSystem.stop()
        self.challengesSubSystem.stop()
        self.kalmanSubSystem.stop()
