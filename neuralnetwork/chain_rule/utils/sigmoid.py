from neuralnetwork.predection.utils.sigmoid import sigmoid


def sigmoid_deriv(x):
    return sigmoid(x) * ( 1 - sigmoid(x) )
