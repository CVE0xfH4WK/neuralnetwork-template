from neuralnetwork.chain_rule.utils.sigmoid import sigmoid_deriv
from neuralnetwork.predection.utils.dot_product import dot_product



def dpredection(ivector, weights, bias):
    predections = {

    }
    
    products = dot_product(
        ivector=ivector,
        weights=weights
    )


    first_layer_1 = products[1] + bias
    second_layer_1 = products[2] + bias

    predections[1] = sigmoid_deriv(
        first_layer_1
    )

    predections[2] = sigmoid_deriv(
        second_layer_1
    )
    

    return predections
