from neuralnetwork.predection.utils.make_predection import predections

def predect(config):
    predections_ = predections(
        config['INPUT_VECTOR'],
        config['WEIGHTS'],
        config['BIAS']
    )

    print(
        f"[1]: {predections_[1]}, [2]: {predections_[2]}"
    )

    return predections_
