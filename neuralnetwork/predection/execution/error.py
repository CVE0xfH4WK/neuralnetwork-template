from neuralnetwork.predection.utils.predection_error import error
from neuralnetwork.predection.utils.make_predection import predections


def _error(derivatives, config):
    new_weigths = [config['WEIGHTS'][0] - derivatives[0], config['WEIGHTS'][1] - derivatives[1]]

    new_predections = predections(
        config['INPUT_VECTOR'],
        new_weigths,
        config['BIAS']
    )

    error_1 = error(
        new_predections[1],
        config['TARGET']
    )

    error_2 = error(
        new_predections[2],
        config['TARGET']
    )

    print(
        f"[1]: {new_predections[1]}, err: {error_1}; [2]: {new_predections[2]}, err: {error_2}"
    )    
