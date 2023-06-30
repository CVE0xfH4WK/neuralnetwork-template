from numpy import random
from numpy import array, dot

from neuralnetwork.predection.utils.sigmoid import sigmoid
from neuralnetwork.predection.utils.predection_error import mse

from neuralnetwork.chain_rule.utils.sigmoid import sigmoid_deriv
from neuralnetwork.chain_rule.utils.predection_error import derror
from neuralnetwork.chain_rule.utils.derror_dbias import derror_dbias



class NET:
    def __init__(self, rate):
        self.weights = array(
            [
                random.randn(),
                random.randn(),
            ]
        )

        self.learning_rate = rate

        self.bias = random.randn()

    
    def predict(
        self, ivector):
        flayer = dot(
            ivector,
            self.weights
        ) + self.bias

        slayer = sigmoid(
            flayer
        )

        return slayer
    

    def compute_gradients(
        self, ivector, 
        target):
        flayer = dot(
            ivector,
            self.weights
        ) + self.bias

        slayer = sigmoid(
            flayer
        )


        derror_ = derror(
            slayer,
            target
        )

        dflayer = sigmoid_deriv(
            flayer
        )

        dflayer_bias = 1

        dfweights = ( 0 * self.weights ) + ( 1 * ivector )

        derror_bias = (
            derror_ * dflayer * dflayer_bias
        )

        derror_weights = (
            derror_ * dflayer * dfweights
        )

        return derror_bias, derror_weights
    

    def update_parameters(
        self, derror_bias,
        derror_weights):
        self.bias = self.bias - ( derror_bias * self.learning_rate )

        self.weights = self.weights - ( derror_weights * self.learning_rate )
    

    def train(
        self, ivectors,
        targets, iterations):
        cumulative_errors = [

        ]

        for iter_ in range(iterations):
            data_index = random.randint(
                len(ivectors)
            )

            ivector = ivectors[data_index]
            target = targets[data_index]
                        

            derr, dweights = self.compute_gradients(
                ivector,
                target
            )

            self.update_parameters(
                derr, dweights
            )


            if iter_ % 100 == 0:
                cumulative_error = 0

                for _data_index in range(len(ivectors)):
                    dpoint = ivectors[_data_index]
                    target_ = targets[_data_index]

                    prediction = self.predict(dpoint)
                    error = mse(prediction, target_)

                    cumulative_error = cumulative_error + error

                cumulative_errors.append(
                    cumulative_error
                )
        
        return cumulative_errors
