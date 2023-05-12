# Container of all kind of random functions needed


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


