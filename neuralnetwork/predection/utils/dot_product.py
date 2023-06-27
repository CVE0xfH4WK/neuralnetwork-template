from numpy import dot


def dot_product(ivector, weights):
    products = {

    }

    products[1] = dot(
        ivector,
        weights[0]
    )

    products[2] = dot(
        ivector,
        weights[1]
    )

    return products
