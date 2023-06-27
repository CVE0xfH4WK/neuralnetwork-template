from neuralnetwork.predection.utils.predection_error import derivative


def _derivative(predections, target):
    derivative_1 = derivative(
            predections[1],
            target
        )

    derivative_2 = derivative(
        predections[2],
        target
    )

    print(
        f'[derivative_1]: {derivative_1}, [derivative_1]: {derivative_2}'
    )

    return [derivative_1, derivative_2]
