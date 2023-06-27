from neuralnetwork.predection.utils.sigmoid import sigmoid
from neuralnetwork.predection.utils.dot_product import dot_product


def predections(ivector, weights, bias):
    predections = {

    }
    
    products = dot_product(
        ivector=ivector,
        weights=weights
    )

    first_layer_1 = products[1] + bias
    second_layer_1 = products[2] + bias

    first_layer_2 = sigmoid(
        first_layer_1
    )
    
    predections[1] = first_layer_2
    
    second_layer_2 = sigmoid(
        second_layer_1
    )

    predections[2] = second_layer_2

    return predections
