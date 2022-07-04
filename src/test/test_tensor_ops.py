#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for testing mathematical operations on tensors.
"""

import abc
from unittest import main, TestCase

import numpy as np

from autograd.tensor import Tensor, add, mul, neg, sub


class _TensorTest(TestCase, abc.ABC):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def _test_values_equality(self, tensor, expected):
        self.assertTrue(
            np.allclose(tensor.data, expected),
            "computed and expected values are not equal"
        )

    def _test_unary_operation(
        self,
        tensor,
        expected_res,
        func,
        tensor_method,
        tensor_operator,
        param=None
    ):
        with self.subTest("stand-alone function"):
            res = func(tensor) if param is None else func(tensor, param)
            self._test_values_equality(res, expected_res)

        with self.subTest("instance method"):
            res = (
                tensor_method()
                if param is None else func(tensor_method(param))
            )
            self._test_values_equality(res, expected_res)

        with self.subTest("unary operator"):
            res = (
                tensor_operator() if param is None else tensor_operator(param)
            )
            self._test_values_equality(res, expected_res)

    def _test_binary_operation(
        self, tensor_1, tensor_2, expected_res, func, tensor_1_method,
        tensor_1_operator
    ):
        with self.subTest("stand-alone function"):
            res = func(tensor_1, tensor_2)
            self._test_values_equality(res, expected_res)

        with self.subTest("instance method"):
            res = tensor_1_method(tensor_2)
            self._test_values_equality(res, expected_res)

        with self.subTest("binary operator"):
            res = tensor_1_operator(tensor_2)
            self._test_values_equality(res, expected_res)

    def _rand_tensor(self, shape):
        return self.rng.random(shape)


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
    def test_add_one_dimensional_no_broadcast(self):
        tensor_1 = Tensor([1, 2, 3], requires_grad=True)
        tensor_2 = Tensor([4, 5, 6], requires_grad=True)

        expected_res = [5, 7, 9]

        self._test_add_operation(tensor_1, tensor_2, expected_res)

    def test_add_multi_dimensional_no_broadcast(self):
        tensor_1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        tensor_2 = Tensor([[4, 5, 6], [7, 8, 9]], requires_grad=True)

        expected_res = [[5, 7, 9], [11, 13, 15]]

        self._test_add_operation(tensor_1, tensor_2, expected_res)

    def test_add_multi_dimensional_with_broadcast(self):
        tensor_1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        tensor_2 = Tensor([4, 5, 6], requires_grad=True)

        expected_res = [[5, 7, 9], [8, 10, 12]]

        self._test_add_operation(tensor_1, tensor_2, expected_res)

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

        grad_1 = upstream_grad
        grad_2 = [-9.0, -18.0, -36.0]

        self._test_values_equality(tensor_1.grad, grad_1)
        self._test_values_equality(tensor_2.grad, grad_2)

    def test_add_backward_with_broadcast_within_tensor(self):
        tensor_1 = Tensor(
            self._rand_tensor((1, 1, 2, 5, 8, 3)), requires_grad=True
        )  # Shape: (1,1,2,5,8,3)
        tensor_2 = Tensor(
            self._rand_tensor((5, 1, 3)), requires_grad=True
        )  # Shape: (5,1,3)

        upstream_grad = np.ones(tensor_1.shape)
        res = tensor_1 + tensor_2
        res.backward(upstream_grad)

        grad_1 = upstream_grad
        grad_2 = np.full(tensor_2.shape, 16)

        self._test_values_equality(tensor_1.grad, grad_1)
        self._test_values_equality(tensor_2.grad, grad_2)

    def _test_add_operation(self, tensor_1, tensor_2, expected_res):
        self._test_binary_operation(
            tensor_1, tensor_2, expected_res, add, tensor_1.add,
            tensor_1.__add__
        )


class TestTensorSub(_TensorTest):
    def test_sub_one_dimensional_no_broadcast(self):
        tensor_1 = Tensor([1, 2, 3], requires_grad=True)
        tensor_2 = Tensor([4, 5, 6], requires_grad=True)

        expected_res = [-3, -3, -3]

        self._test_sub_operation(tensor_1, tensor_2, expected_res)

    def test_sub_multi_dimensional_no_broadcast(self):
        tensor_1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        tensor_2 = Tensor([[6, 5, 4], [9, 8, 7]], requires_grad=True)

        expected_res = [[-5, -3, -1], [-5, -3, -1]]

        self._test_sub_operation(tensor_1, tensor_2, expected_res)

    def test_sub_multi_dimensional_with_broadcast(self):
        tensor_1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        tensor_2 = Tensor([4, 5, 6], requires_grad=True)

        expected_res = [[-3, -3, -3], [0, 0, 0]]

        self._test_sub_operation(tensor_1, tensor_2, expected_res)

    def test_sub_backward_no_broadcast(self):
        tensor_1 = Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        tensor_2 = Tensor(
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
        )

        upstream_grad = [[-1.0, 2.0, -4.0], [8.0, -16.0, 32.0]]
        res = tensor_1 - tensor_2
        res.backward(upstream_grad)

        grad_1 = upstream_grad
        grad_2 = [[1.0, -2.0, 4.0], [-8.0, 16.0, -32.0]]

        self._test_values_equality(tensor_1.grad, grad_1)
        self._test_values_equality(tensor_2.grad, grad_2)

    def test_sub_backward_with_broadcast_prepend_dims_only(self):
        tensor_1 = Tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True
        )  # Shape: (1,2,3)
        tensor_2 = Tensor([4.0, 5.0, 6.0], requires_grad=True)  # Shape: (3,)

        upstream_grad = [[[-1.0, 2.0, -4.0], [8.0, -16.0, 32.0]]]
        res = tensor_1 - tensor_2
        res.backward(upstream_grad)

        grad_1 = upstream_grad
        grad_2 = [-7.0, 14.0, -28.0]

        self._test_values_equality(tensor_1.grad, grad_1)
        self._test_values_equality(tensor_2.grad, grad_2)

    def test_sub_backward_with_broadcast_within_tensor(self):
        tensor_1 = Tensor(
            self._rand_tensor((1, 1, 2, 5, 8, 3)), requires_grad=True
        )  # Shape: (1,1,2,5,8,3)
        tensor_2 = Tensor(
            self._rand_tensor((5, 1, 3)), requires_grad=True
        )  # Shape: (5,1,3)

        upstream_grad = np.ones(tensor_1.shape)
        res = tensor_1 - tensor_2
        res.backward(upstream_grad)

        grad_1 = upstream_grad
        grad_2 = np.full(tensor_2.shape, -16)

        self._test_values_equality(tensor_1.grad, grad_1)
        self._test_values_equality(tensor_2.grad, grad_2)

    def _test_sub_operation(self, tensor_1, tensor_2, expected_res):
        self._test_binary_operation(
            tensor_1, tensor_2, expected_res, sub, tensor_1.sub,
            tensor_1.__sub__
        )


class TestTensorMul(_TensorTest):
    def test_mul_func_one_dimensional_no_broadcast(self):
        tensor_1 = Tensor([1, 2, 3], requires_grad=True)
        tensor_2 = Tensor([4, 5, 6], requires_grad=True)

        expected_res = [4, 10, 18]

        self._test_mul_operation(tensor_1, tensor_2, expected_res)

    def test_mul_func_multi_dimensional_no_broadcast(self):
        tensor_1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        tensor_2 = Tensor([[4, 5, 6], [7, 8, 9]], requires_grad=True)

        expected_res = [[4, 10, 18], [28, 40, 54]]

        self._test_mul_operation(tensor_1, tensor_2, expected_res)

    def test_mul_func_multi_dimensional_with_broadcast(self):
        tensor_1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        tensor_2 = Tensor([4, 5, 6], requires_grad=True)

        expected_res = [[4, 10, 18], [16, 25, 36]]

        self._test_mul_operation(tensor_1, tensor_2, expected_res)

    def test_mul_backward_no_broadcast(self):
        data_1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        data_2 = [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

        tensor_1 = Tensor(data_1, requires_grad=True)
        tensor_2 = Tensor(data_2, requires_grad=True)

        upstream_grad = [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]

        res = tensor_1 * tensor_2
        res.backward(upstream_grad)

        grad_1 = [[-4.0, -10.0, -18.0], [-28.0, -40.0, -54.0]]
        grad_2 = [[-1.0, -4.0, -9.0], [-16.0, -25.0, -36.0]]

        self._test_values_equality(tensor_1.grad, grad_1)
        self._test_values_equality(tensor_2.grad, grad_2)

    def test_mul_backward_with_broadcast_prepend_dims_only(self):
        data_1 = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        data_2 = [4.0, 5.0, 6.0]

        tensor_1 = Tensor(data_1, requires_grad=True)  # Shape: (1,2,3)
        tensor_2 = Tensor(data_2, requires_grad=True)  # Shape: (3,)

        upstream_grad = [[[-1.0, -2.0, -4.0], [-8.0, -16.0, -32.0]]]

        res = tensor_1 * tensor_2
        res.backward(upstream_grad)

        grad_1 = [[-4.0, -10.0, -24.0], [-32.0, -80.0, -192.0]]
        grad_2 = [-33.0, -84.0, -204.0]

        self._test_values_equality(tensor_1.grad, grad_1)
        self._test_values_equality(tensor_2.grad, grad_2)

    def test_mul_backward_with_broadcast_within_tensor(self):
        data_1 = self._rand_tensor((1, 1, 2, 5, 8, 3))
        data_2 = self._rand_tensor((5, 1, 3))

        tensor_1 = Tensor(data_1, requires_grad=True)  # Shape: (1,1,2,5,8,3)
        tensor_2 = Tensor(data_2, requires_grad=True)  # Shape: (5,1,3)

        res = tensor_1 * tensor_2
        upstream_grad = np.ones(res.shape)
        res.backward(upstream_grad)

        grad_1 = upstream_grad * data_2
        grad_2 = np.sum(
            upstream_grad * data_1, axis=(0, 1, 2, 4), keepdims=True
        )

        self._test_values_equality(tensor_1.grad, grad_1)
        self._test_values_equality(tensor_2.grad, grad_2)

    def _test_mul_operation(self, tensor_1, tensor_2, expected_res):
        self._test_binary_operation(
            tensor_1, tensor_2, expected_res, mul, tensor_1.mul,
            tensor_1.__mul__
        )


class TestTensorNeg(_TensorTest):
    def test_simple(self):
        tensor = Tensor([1, 2, 3, 4])
        expected_res = [-1, -2, -3, -4]

        self._test_neg_operation(tensor, expected_res)

    def test_complex(self):
        tensor = Tensor([[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]])
        expected_res = [[-1.0, 2.0, -3.0, 4.0], [-5.0, 6.0, -7.0, 8.0]]

        self._test_neg_operation(tensor, expected_res)

    def test_neg_backward(self):
        tensor = Tensor(
            [[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]],
            requires_grad=True
        )
        res = -tensor

        upstream_grad = [[1, 1, 1, 1], [1, 1, 1, 1]]
        res.backward(upstream_grad)

        grad = [[-1, -1, -1, -1], [-1, -1, -1, -1]]

        self._test_values_equality(tensor.grad, grad)

    def _test_neg_operation(self, tensor, expected_res):
        self._test_unary_operation(
            tensor, expected_res, neg, tensor.neg, tensor.__neg__
        )


if __name__ == '__main__':
    main()
