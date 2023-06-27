from neuralnetwork.chain_rule.utils.make_predection import dpredection

def predect(config):
    predections_ = dpredection(
        config['INPUT_VECTOR'],
        config['WEIGHTS'],
        config['BIAS']
    )

    print(
        f"[1]: {predections_[1]}, [2]: {predections_[2]}"
    )

    return predections_
