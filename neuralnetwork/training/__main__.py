from numpy import array

from config.reader import read
from argparser.parser import parser

from neuralnetwork.training.utils.net import NET
from neuralnetwork.training.utils.plot import save_graf


def main():
    args = parser()

    config = read(args.config)


    ivectors = array(config["VEC"])

    targets = array(config["TARS"])


    model = NET(
        config["RATE"]
    )

    model_errors = model.train(
        ivectors,
        targets,
        config["ITERS"]
    )


    save_graf(
        model_errors
    )


if __name__ == '__main__':
    main()
