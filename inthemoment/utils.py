# utils.py
# Authors Stephan Meighen-Berger
# Utility functions for the inthemoment package

import numpy as np


def find_nearest(array, value):
    """ Function to find the closest value in the array

    Parameters
    ----------
    array: iterable
        Iterable object
    value: float/int
        The value to find the closest element for

    Returns
    -------
    idx: int
        The index of the closest value
    """
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
