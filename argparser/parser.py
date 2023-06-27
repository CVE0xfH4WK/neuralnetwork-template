from argparse import ArgumentParser


def parser():
    parser = ArgumentParser()

    parser.add_argument(
        '-c',
        '--config'
    )

    return parser.parse_args()
