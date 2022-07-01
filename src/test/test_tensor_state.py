#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for testing tensor state-related methods.
"""

from unittest import main, TestCase

import numpy as np

from autograd.tensor import Tensor
from autograd.exceptions import InvalidGradient, InvalidTensorState


class TestTensorState(TestCase):
    def test_data_equality(self):
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        tensor = Tensor(data)

        self.assertTrue(np.allclose(tensor.data, data))

    def test_shape_scalar(self):
        self._test_shape(5, ())

    def test_shape_one_dimensional(self):
        self._test_shape([1, 2, 3, 4, 5, 6, 7], (7, ))

    def test_shape_two_dimensional(self):
        self._test_shape([[1, 2, 3], [4, 5, 6]], (2, 3))

    def test_shape_three_dimensional(self):
        self._test_shape([[[1], [2], [3]], [[4], [5], [6]]], (2, 3, 1))

    def test_gradient_is_none_by_default(self):
        tensor = Tensor(1.0)

        self.assertIsNone(tensor.grad, "gradient has to be None by default")

    def test_gradient_is_not_none_after_backward(self):
        tensor = Tensor(1.0, requires_grad=True)
        tensor.backward()

        self.assertIsNotNone(
            tensor.grad, "gradient must not be None after backward"
        )

    def test_zero_gradient_contains_float_zeroes(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        grad = [[0.0, 0.0], [0.0, 0.0]]

        tensor = Tensor(data)
        tensor.zero_grad()

        self.assertTrue(
            np.allclose(tensor.grad.data, grad),
            "gradient has to contain only zeroes after clear operation"
        )

    def test_zero_gradient_has_equal_shape_to_data_shape(self):
        data = [[[1.0], [2.0]], [[3.0], [4.0]]]

        tensor = Tensor(data)
        tensor.zero_grad()

        self.assertEqual(
            tensor.grad.data.shape, (2, 2, 1),
            "shape of the gradients needs to match the shape of the data"
        )

    def test_invalid_state_if_not_requires_grad_before_backward(self):
        with self.assertRaises(
            InvalidTensorState,
            msg="tensor needs to require gradients if backpropagation is called"
        ):
            tensor = Tensor([[1, 2], [3, 4]], requires_grad=False)
            tensor.backward()

    def test_invalid_gradient_if_grad_not_specified_for_non_scalar_tensor(self):
        with self.assertRaises(
            InvalidGradient,
            msg="it is impossible to compute a gradient for non-scalar tensor"
            "without specifying the upstream gradient"
        ):
            tensor = Tensor([[5, 6], [7, 8], [9, 10]], requires_grad=True)
            tensor.backward()

    def _test_shape(self, data, expected_shape):
        tensor = Tensor(data)

        self.assertEqual(
            tensor.shape, expected_shape, "tensor shapes do not match"
        )


if __name__ == '__main__':
    main()
