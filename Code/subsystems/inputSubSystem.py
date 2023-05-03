import robot
from misc.robotModeEnum import robotMode
from misc.robotStatusEnum import robotStatus
from subsystems.subsystem import subSystem
from subsystems.subsystemStateEnum import subSystemState


# handles manual control mode

class inputSubSystem(subSystem):
    enabled = False  # used internally
    powerSensitivity = 2  # tune this for a quicker or slower response to W/S inputs. Should always be a INT
    turningSensitivity = 4  # tune this for a quicker or slower response to A/D inputs. Should always be a INT

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
            _new_power = robot.Robot.input_motor + self.powerSensitivity
            robot.Robot.input_motor = max(135, min(165,_new_power))

    def keyboard_s(self):
        if self.enabled:
            _new_power = robot.Robot.input_motor - self.powerSensitivity
            robot.Robot.input_motor = max(135, min(165, _new_power))

    def keyboard_a(self):
        if self.enabled:
            _new_servo = robot.Robot.input_servo - self.turningSensitivity
            robot.Robot.input_servo = max(100, min(200, _new_servo))

    def keyboard_d(self):
        if self.enabled:
            _new_servo = robot.Robot.input_servo + self.turningSensitivity
            robot.Robot.input_servo = max(100, min(200, _new_servo))

    def estop(self):
        # TODO: possibly implement a warning sound when the estop is pressed?
        """
        Used for stopping everything in case of an emergency. Called via a button in the GUI or space on the keyboard
        @return:
        """
        print("ESTOP")
        robot.Robot.operatingMode = robotMode.EStop
        robot.Robot.status = robotStatus.ESTOP
        robot.Robot.input_servo = 150
        robot.Robot.input_motor = 150

        self.enabled = False

    def update(self):
        robot.Robot.inputState = self.state
