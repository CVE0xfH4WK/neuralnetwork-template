from config.reader import read
from argparser.parser import parser

from neuralnetwork.predection.execution.mse import _mse
from neuralnetwork.predection.execution.error import _error
from neuralnetwork.predection.execution.predection import predect
from neuralnetwork.predection.execution.derivative import _derivative


def main():
    args = parser()

    config = read(args.config)

    predections = predect(
        config
    )
    
    _mse(
        predections,
        config['TARGET']
    )

    derivatives = _derivative(
        predections,
        config['TARGET']
    )

    _error(
        derivatives,
        config 
    )


if __name__ == '__main__':
    main()
