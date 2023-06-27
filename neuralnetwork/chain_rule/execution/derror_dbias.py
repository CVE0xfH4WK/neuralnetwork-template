from neuralnetwork.chain_rule.utils.derror_dbias import derror_dbias


def deb(errors, predections, dbias=1):
    deb_1 = derror_dbias(
        errors[0],
        predections[1],
        dbias
    )

    deb_2 = derror_dbias(
        errors[1],
        predections[2],
        dbias
    )

    print(
        f'[deb_1]: {deb_1}; [deb_2]: {deb_2}'
    )
