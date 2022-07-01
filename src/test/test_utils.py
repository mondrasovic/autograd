#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that contains tests for utility functions."""

from unittest import main, TestCase

from autograd.utils import are_broadcastable


class TestAreBroadcastableFunction(TestCase):
    def test_simple_true(self):
        self.assertTrue(are_broadcastable((3, 2, 5), (5, ), (2, 5)))

    def test_simple_false(self):
        self.assertFalse(are_broadcastable((4, 5), (5, 4)))

    def test_complex_true(self):
        self.assertTrue(
            are_broadcastable(
                (5, 4, 7, 3, 2), (4, 1, 3, 2), (3, 2), (2, ), (1, )
            )
        )

    def test_complex_false(self):
        self.assertFalse(are_broadcastable((1, 2, 3, 4, 5), (5, 4, 3, 2, 1)))


if __name__ == '__main__':
    main()