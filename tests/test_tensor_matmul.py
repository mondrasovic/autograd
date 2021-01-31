import unittest

import numpy as np

from autograd.tensor import Tensor

class TestTensorMatMul(unittest.TestCase):
    def test_simple_matmul(self):
        # t1 is (3, 2)
        t1 = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
        # t2 is a (2, 1)
        t2 = Tensor([[10], [20]], requires_grad=True)

        # t3 is (3, 1)
        t3 = t1 @ t2

        self.assertEqual(t3.data.tolist(), [[50], [110], [170]])

        grad = Tensor([[-1], [-2], [-3]])
        t3.backward(grad)

        self.assertEqual(
            t1.grad.data.tolist(), (grad.data @ t2.data.T).tolist())
        self.assertEqual(
            t2.grad.data.tolist(), (t1.data.T @ grad.data).tolist())
