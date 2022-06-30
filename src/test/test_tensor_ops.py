#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
from unittest import main, TestCase

import numpy as np

from autograd.tensor import Tensor, add


class _TensorTest(TestCase, abc.ABC):
    def _test_values_equality(self, tensor, expected):
        with self.subTest('values equality (floating-point tolerance)'):
            self.assertTrue(
                np.allclose(tensor.data, expected),
                "computed and expected values are not equal"
            )

    def _test_binary_operation(
        self, tensor_1, tensor_2, expected_res, func, tensor_1_method,
        tensor_1_operator
    ):
        with self.subTest('stand-alone function'):
            res = func(tensor_1, tensor_2)
            self._test_values_equality(res, expected_res)

        with self.subTest('instance method'):
            res = tensor_1_method(tensor_2)
            self._test_values_equality(res, expected_res)

        with self.subTest('binary operator'):
            res = tensor_1_operator(tensor_2)
            self._test_values_equality(res, expected_res)


class TestTensorSum(_TensorTest):
    def test_reduce_sum(self):
        tensor = Tensor([[1, 2], [3, 4], [5, 6]])
        self._test_values_equality(tensor.reduce_sum(), 21)

    def test_reduce_sum_backward(self):
        data = np.asarray([[1, 2], [3, 4], [5, 6]])
        grad_value_pairs = ((None, 1), (1, 1), (2, 2), (10, 10))

        for upstream_grad, expected_grad_value in grad_value_pairs:
            with self.subTest(upstream_grad):
                tensor = Tensor(data, requires_grad=True)

                tensor_sum = tensor.reduce_sum()
                tensor_sum.backward(upstream_grad)

                self._test_values_equality(
                    tensor_sum.grad, np.full_like(data, expected_grad_value)
                )


class TestTensorAdd(_TensorTest):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_add_func_one_dimensional_no_broadcast(self):
        tensor_1 = Tensor([1, 2, 3], requires_grad=True)
        tensor_2 = Tensor([4, 5, 6], requires_grad=True)

        expected_res = [5, 7, 9]

        self._test_binary_operation(
            tensor_1, tensor_2, expected_res, add, tensor_1.add,
            tensor_1.__add__
        )

    def test_add_func_multi_dimensional_no_broadcast(self):
        tensor_1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        tensor_2 = Tensor([[4, 5, 6], [7, 8, 9]], requires_grad=True)

        expected_res = [[5, 7, 9], [11, 13, 15]]

        self._test_binary_operation(
            tensor_1, tensor_2, expected_res, add, tensor_1.add,
            tensor_1.__add__
        )

    def test_add_func_multi_dimensional_with_broadcast(self):
        tensor_1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        tensor_2 = Tensor([4, 5, 6], requires_grad=True)

        expected_res = [[5, 7, 9], [8, 10, 12]]

        self._test_binary_operation(
            tensor_1, tensor_2, expected_res, add, tensor_1.add,
            tensor_1.__add__
        )

    def test_add_backward_no_broadcast(self):
        tensor_1 = Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        tensor_2 = Tensor(
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
        )

        upstream_grad = [[-1.0, -2.0, -4.0], [-8.0, -16.0, -32.0]]
        res = tensor_1 + tensor_2
        res.backward(upstream_grad)

        self._test_values_equality(tensor_1.grad, upstream_grad)
        self._test_values_equality(tensor_2.grad, upstream_grad)

    def test_add_backward_with_broadcast_prepend_dims_only(self):
        tensor_1 = Tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True
        )  # Shape: (1,2,3)
        tensor_2 = Tensor([4.0, 5.0, 6.0], requires_grad=True)  # Shape: (3,)

        upstream_grad = [[[-1.0, -2.0, -4.0], [-8.0, -16.0, -32.0]]]
        res = tensor_1 + tensor_2
        res.backward(upstream_grad)

        grad_tensor_1 = upstream_grad
        grad_tensor_2 = [-9.0, -18.0, -36.0]

        self._test_values_equality(tensor_1.grad, grad_tensor_1)
        self._test_values_equality(tensor_2.grad, grad_tensor_2)

    def test_add_backward_with_broadcast_within_tensor(self):

        tensor_1 = Tensor(
            self.rng.random((1, 1, 2, 5, 8, 3)), requires_grad=True
        )  # Shape: (1,1,2,5,8,3)
        tensor_2 = Tensor(
            self.rng.random((5, 1, 3)), requires_grad=True
        )  # Shape: (5,1,3)

        upstream_grad = np.ones(tensor_1.shape)
        res = tensor_1 + tensor_2
        res.backward(upstream_grad)

        grad_tensor_1 = upstream_grad
        grad_tensor_2 = np.full(tensor_2.shape, 16)

        self._test_values_equality(tensor_1.grad, grad_tensor_1)
        self._test_values_equality(tensor_2.grad, grad_tensor_2)


if __name__ == '__main__':
    main()
