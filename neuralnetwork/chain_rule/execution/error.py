from neuralnetwork.chain_rule.utils.predection_error import derror


def _error(predections, config):
    error_1 = derror(
        predections[1],
        config['TARGET']
    )

    error_2 = derror(
        predections[2],
        config['TARGET']
    )

    print(
        f"[1]: {predections[1]}, err: {error_1}; [2]: {predections[2]}, err: {error_2}"
    )    

    return [error_1, error_2]
