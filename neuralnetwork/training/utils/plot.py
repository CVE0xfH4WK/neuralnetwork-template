import matplotlib.pyplot as plot


def save_graf(errs):
    plot.plot(
        errs
    )

    plot.xlabel(
        "Iterations"
    )

    plot.ylabel(
        "Error for all training instances"
    )

    plot.savefig(
        "cumulative_errs.png"
    )
