# Container of all kind of random functions needed
import numpy as np

import robot


# . Returns true or false

def ish(value: float, check: float, marginOfError: float) -> bool:
    """
    used for checking if a value is close to another value
    @param value: The value you want to check. E.G. a sensor value
    @param check: The value you want to check against. E.G. a setpoint
    @param marginOfError: The margin of error for the value
    @return: returns a boolean
    """
    if value > check + marginOfError:  # So the value we want to check is larger than the check value + the margin of error, so it is not approximatly equal to the check value
        return False
    elif value < check - marginOfError:  # So the value we want to check is smaller than the check value - the margin of error, so it is not approximatly equal to the check value
        return False
    else:
        return True

def steer_to_angle(input: float, type: str) -> float:
    """takes the steering input and returns the angle in radian or degree

    Args:
        input (float): steering input 
        type (str): type of output required, either radian or degree

    Returns:
        float: steering input as either degrees or radians
    """
    if type == "radian":
        radian = (0.0077*input)-1.1549
        return radian
    elif type == "degree":
        degree = (0.4432*input)-66.168
        if np.absolute(degree) < 0.5:
            degree = 0 #added for small offset in model
        return degree
    else:
        return print("give valid type - radian or degree")

def motor_to_force(input: float) -> float:
    """
    Transforms the motor input to a force (equal to Fa - Fb)
    @param input: between 135 and 165
    @return: force in newtons
    """
    if(146 < input < 156):
        return 0
    if(input >= 156):
        return (input-150)/15*robot.Robot.faMax
    if (input <= 146):
        return (150-input) / 15 * robot.Robot.fbMax

