from neuralnetwork.predection.utils.predection_error import mse

def _mse(predections, target):
    
    mse_1 = mse(
        predections[1],
        target
    )

    mse_2 = mse(
        predections[2],
        target
    )

    print(
        f"[1]: {predections[1]}, mse: {mse_1}; [2]: {predections[2]}, mse: {mse_2}"
    )
