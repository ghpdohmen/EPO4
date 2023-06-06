from misc.robotModeEnum import robotMode


#TODO: rewrite as a switch statement
def robotModeParser(i):
    """
    Parses an int to a robotmode
    @param i: int index
    @return: robotmode
    """
    match i:
        case 0:
            return robotMode.Manual
        case 1:
            return robotMode.ChallengeA
        case 2:
            return robotMode.ChallengeB
        case 3:
            return robotMode.ChallengeC
        case 4:
            return robotMode.ChallengeD
        case 5:
            return robotMode.ChallengeE
        case 7:
            return robotMode.EStop
    #if there is no matching case, then return Notchosen
    return robotMode.NotChosen
