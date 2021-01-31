import unittest

import numpy as np

from autograd.tensor import Tensor

class TestTensorAdd(unittest.TestCase):
    def test_simple_add(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 + t2

        self.assertEqual(t3.data.tolist(), [5, 7, 9])

        t3.backward(Tensor([-1, -2, -3]))

        self.assertEqual(t1.grad.data.tolist(), [-1, -2, -3])
        self.assertEqual(t2.grad.data.tolist(), [-1, -2, -3])
    
    def test_add_in_place(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t1 += 0.1

        self.assertIsNone(t1.grad)
        self.assertTrue(np.allclose(t1.data, [1.1, 2.1, 3.1]))
    
    def test_broadcast_add1(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([7, 8, 9], requires_grad=True)  # (3,)

        t3 = t1 + t2  # (2, 3)

        self.assertEqual(t3.data.tolist(), [[8, 10, 12], [11, 13, 15]])

        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        self.assertEqual(t1.grad.data.tolist(), [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(t2.grad.data.tolist(), [2, 2, 2])
    
    def test_broadcast_add2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=True)  # (1, 3)

        t3 = t1 + t2  # (2, 3)

        self.assertEqual(t3.data.tolist(), [[8, 10, 12], [11, 13, 15]])

        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        self.assertEqual(t1.grad.data.tolist(), [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(t2.grad.data.tolist(), [[2, 2, 2]])

