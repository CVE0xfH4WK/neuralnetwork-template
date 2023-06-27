from config.reader import read
from argparser.parser import parser

from neuralnetwork.chain_rule.execution.error import _error
from neuralnetwork.chain_rule.execution.derror_dbias import deb
from neuralnetwork.chain_rule.execution.predection import predect


def main():
    args = parser()

    config = read(args.config)

    predections = predect(
        config
    )

    errors = _error(
        predections,
        config
    )

    deb(
        errors,
        predections,
    )


if __name__ == '__main__':
    main()
