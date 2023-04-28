import robot
from misc.robotModeEnum import robotMode
from misc.robotStatusEnum import robotStatus
from subsystems.subsystem import subSystem
from subsystems.subsystemStateEnum import subSystemState


# handles manual control mode

class inputSubSystem(subSystem):
    enabled = False  # used internally
    powerSensitivity = 1  # tune this for a quicker or slower response to W/S inputs. Should always be a INT
    turningSensitivity = 1  # tune this for a quicker or slower response to A/D inputs. Should always be a INT

    def start(self):
        self.state = subSystemState.Started
        robot.Robot.inputState = self.state
        if robot.Robot.operatingMode == robotMode.Manual:
            self.enabled = True
            self.state = subSystemState.Running
            robot.Robot.status = robotStatus.Running

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.inputState = self.state
        robot.Robot.status = robotStatus.Completed
        self.enabled = False

    def keyboard_w(self):
        if self.enabled:
            robot.Robot.input_motor += self.powerSensitivity

    def keyboard_s(self):
        if self.enabled:
            robot.Robot.input_motor -= self.powerSensitivity

    def keyboard_a(self):
        if self.enabled:
            robot.Robot.input_servo -= self.turningSensitivity

    def keyboard_d(self):
        if self.enabled:
            robot.Robot.input_servo -= self.turningSensitivity

    def estop(self):
        # TODO: decide on a key for the emergency stop
        # TODO: possibly implement a warning sound when the estop is pressed?
        """
        Used for stopping everything in case of an emergency. Called via a button in the GUI or ... on the keyboard
        @return:
        """
        robot.Robot.operatingMode = robotMode.EStop
        robot.Robot.status = robotStatus.EStop
        self.enabled = False
        robot.Robot.input_servo = 150
        robot.Robot.input_motor = 150

    def update(self):
        robot.Robot.inputState = self.state
