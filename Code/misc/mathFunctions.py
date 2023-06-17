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
        radian = (-0.0077 * input) + 1.1549
        if np.abolute(radian) < 0.01:  #TODO: check whether this is okay
            radian = 0
        return radian
    elif type == "degree":
        degree = (-0.4432 * input) + 66.168
        if np.absolute(degree) < 0.5:
            degree = 0  # added for small offset in model
        return degree
    else:
        return print("give valid type - radian or degree")


def angle_to_steer(input: float) -> int:
    """
    takes an angle and returns the steering input for the KITT robot (int)
    :param input: angle in degrees
    :return: steering input for KITT
    """
    steering_input = (input - 66.168)/(-0.4432)
    if steering_input > 200:  # makes sure the returned input is within boundaries
        steering_input = 200
        return steering_input

    elif steering_input < 100:
        steering_input = 100
        return steering_input

    else:
        return round(steering_input)


def motor_to_force(input: float) -> float:
    """
    Transforms the motor input to a force (equal to Fa - Fb)
    @param input: between 135 and 165
    @return: force in newtons
    """
    if (146 < input < 156):
        return 0
    if (input >= 156):
        return (input - 150) / 15 * robot.Robot.faMax
    if (input <= 146):
        return (150 - input) / 15 * robot.Robot.fbMax


def distance_calc(x2, y2, x1, y1):
    """
    gives the distance between two points
    :param x2:
    :param y2:
    :param x1:
    :param y1:
    :return:
    """
    _distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return _distance


def which_one_is_closer(point_1, point_2, destination):
    """
    gives the point closer to the destination point out of two other points
    :param point_1:
    :param point_2:
    :param destination:
    :return:
    """
    _distance_1 = distance_calc(point_1[1], point_1[0], destination[1], destination[0])
    _distance_2 = distance_calc(point_2[1], point_2[0], destination[1], destination[0])

    if _distance_1 <= _distance_2:
        return point_1
    else:
        return point_2

