from misc.robotModeEnum import robotMode


#TODO: rewrite as a switch statement
def robotModeParser(i):
    """
    Parses an int to a robotmode
    @param i: int index
    @return: robotmode
    """
    if i == 0:
        return robotMode.Manual
    elif i == 1:
        return robotMode.ChallengeA
    elif i == 2:
        return robotMode.ChallengeB
    elif i == 3:
        return robotMode.ChallengeC
    elif i == 4:
        return robotMode.ChallengeD
    elif i == 5:
        return robotMode.ChallengeE
    elif i == 7:
        return robotMode.EStop
    else:
        return robotMode.NotChosen
