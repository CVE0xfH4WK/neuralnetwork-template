from numpy import square


def mse(predection, target):
    return square(
        predection - target
    )


def derivative(predection, target):
    return 2 * ( predection - target )


def error(new_predection, target):
    return ( new_predection - target ) ** 2
