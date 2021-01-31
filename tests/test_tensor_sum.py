import unittest

import numpy as np

from autograd.tensor import Tensor

class TestTensorSum(unittest.TestCase):
    def test_simple_sum(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1.sum()

        self.assertEqual(t2.data.tolist(), 6)

        t2.backward()

        self.assertEqual(t1.grad.data.tolist(), [1, 1, 1])
