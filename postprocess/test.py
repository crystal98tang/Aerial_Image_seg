import numpy


def dc(input1, input2):
    """
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC = \frac{2|A\capB|}{|A|+|B|}

    , where A is the first and B the second set of samples (here binary objects).

    Parameters
    ----------
    input1: array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    input2: array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc: float
        The Dice coefficient between the object(s) in `input1` and the
        object(s) in `input2`. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric.
    """
    input1 = numpy.atleast_1d(input1.astype(numpy.bool))
    input2 = numpy.atleast_1d(input2.astype(numpy.bool))

    intersection = numpy.count_nonzero(input1 & input2)

    size_i1 = numpy.count_nonzero(input1)
    size_i2 = numpy.count_nonzero(input2)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


