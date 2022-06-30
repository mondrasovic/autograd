#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that contains custom types used within this library.
"""

import numbers

from typing import Callable, Sequence, Tuple, Union

import numpy as np

TensorableT = Union[numbers.Number, np.ndarray, Sequence]
# It's better to base the gradient function on top of NumPy arrays instead of
# tensors to reduce boilerplate code.
GradFnT = Callable[[np.ndarray], np.ndarray]
ShapeT = Tuple[int, ...]
