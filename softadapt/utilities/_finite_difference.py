"""Internal implementation of """
import numpy
from findiff import coefficients
from ..constants._finite_difference_constants import (_FIRST_ORDER_COEFFICIENTS,
                        _THIRD_ORDER_COEFFICIENTS, _FIFTH_ORDER_COEFFICIENTS)


def _get_finite_difference(input_array: numpy.array,
                           order: int = None,
                           verbose: bool = True):
    """Internal utility method for estimating rate of change.

    This function aims to approximate the rate of change for a loss function,
    which is used for the 'LossWeighted' and 'Normalized' variants of SoftAdapt.

    For even accuracy orders, we take advantage of the `findiff` package
    (https://findiff.readthedocs.io/en/latest/source/examples-basic.html).
    Accuracy orders of 1 (trivial), 3, and 5 are retrieved from an internal
    constants file. Due to the underlying mathematics of computing the
    coefficients, all accuracy orders higher than 5 must be an even number.

    Args:
        input_array: An array of floats containing loss evaluations at the
          previous 'n' points (as many points as the order) of the finite
          difference method.
        order: An integer indicating the order of the finite difference method
          we want to use. The function will use the length of the 'input_array'
          array if no values is provided.
        verbose: Whether we want the function to print out information about
          computations or not.

    Returns:
        A float which is the approximated rate of change between the loss
        points.

    Raises:
        ValueError: If the number of points in the `input_array` array is
          smaller than the order of accuracy we desire.
        Value Error: If the order of accuracy is higher than 5 and it is not an
          even number.
    """
    # First, we want to check the order and the number of loss points we are
    # given
    if order is None:
        order = len(input_array) - 1
        if verbose:
            print(f"==> Interpreting finite difference order as {order} since"
                  "no explicit order was specified.")
    else:
        if order > len(input_array):
            raise ValueError("The order of finite difference computations can"
                             "not be larger than the number of loss points. "
                             "Please check the order argument or wait until "
                             "enough points have been stored before calling the"
                             " method.")
        elif order + 1 < len(input_array):
            print(f"==> There are more points than 'order' + 1 ({order + 1}) "
                  f"points (array contains {len(input_array)} values). Function"
                  f"will use the last {order} elements of loss points for "
                  "computations.")
            input_array = input_array[(-1*order - 1):]

    order_is_even = order % 2 == 0
    # Next, we want to retrieve the correct coefficients based on the order
    if order > 5 and not order_is_even:
        raise ValueError("Accuracy orders larger than 5 must be even. Please "
                         "check the arguments passed to the function.")

    if order_is_even:
        constants = coefficients(deriv=1, acc=order)["forward"]["coefficients"]

    else:
        if order == 1:
            constants = _FIRST_ORDER_COEFFICIENTS
        elif order == 3:
            constants = _THIRD_ORDER_COEFFICIENTS
        else:
            constants = _FIFTH_ORDER_COEFFICIENTS

    pointwise_multiplication = [
        input_array[i] * constants[i] for i in range(len(constants))
    ]
    return numpy.sum(pointwise_multiplication)
