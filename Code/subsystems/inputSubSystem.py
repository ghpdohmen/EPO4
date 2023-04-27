import robot
from misc.robotModeEnum import robotMode
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

    def stop(self):
        self.state = subSystemState.Stopped
        robot.Robot.inputState = self.state
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

    def update(self):
        robot.Robot.inputState = self.state
