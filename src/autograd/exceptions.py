#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module containing a base class for all the library-related exceptions.
"""

__all__ = ['AutogradError']


class AutogradError(Exception):
    """Base class for all exceptions raised within this library."""
    pass


class InvalidTensorState(AutogradError):
    """An exception raised to indicate that some operation was called upon a
    tensor that does not meet the requirements in terms of its state.
    """
    pass


class InvalidGradient(AutogradError):
    """An exception indicating that an invalid gradient parameter has been
    passed during backpropagation phase.
    """
    pass