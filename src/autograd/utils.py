#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that contains auxiliary utility functions."""

import numpy as np

from typing import Tuple


def are_broadcastable(*args: Tuple[Tuple[int, ...], ...]) -> bool:
    """Determines whether the provided tensor shapes adhere to the rules of
    broadcasting.

    Args:
        `*args` (Tuple[Tuple[int, ...], ...]): Tuple of shapes, where each shape
            is a tuple of ints.

    Returns:
        bool: True, if the specified shapes are broadcastable, False otherwise.
    """
    try:
        np.broadcast_shapes(*args)
    except ValueError:
        return False
    else:
        return True
