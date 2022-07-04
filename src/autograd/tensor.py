#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that contains the basic tensor class together with related classes.
"""

import dataclasses

import numpy as np

from typing import Sequence, Optional, Iterator, Union

from autograd.exceptions import InvalidTensorState, InvalidGradient
from autograd.types import GradFnT, TensorableT, ShapeT
from autograd.utils import are_broadcastable


@dataclasses.dataclass(frozen=True)
class Dependency:
    """Dependency between two tensors.
    
    A dependency points back to some other tensor. The information about
    gradient function is needed for the backpropagation phase. It represents an
    edge within the computational graph.
    """
    tensor: 'Tensor'
    grad_fn: GradFnT


class Tensor:
    """Elemental class representing a basic tensor.
    """
    def __init__(
        self,
        data: TensorableT,
        requires_grad: bool = False,
        dependencies: Optional[Sequence[Dependency]] = None
    ) -> None:
        """Creates a new tensor instance.

        Args:
            data (TensorableT): Underlying data representing tensor content.
            requires_grad (bool, optional): Indicates whether the tensor
                requires gradient or not. Defaults to False.
            dependencies (Optional[Sequence[Dependency]], optional): A list of
                dependencies (tensor with accompanying gradient functions).
                Defaults to None.
        """
        self._data = np.asarray(data)
        self.requires_grad = requires_grad
        self.dependencies = dependencies
        self.grad: Optional['Tensor'] = None

    def __len__(self) -> int:
        """Returns the length of the tensor, i.e., the size of the first 
        dimension.

        Returns:
            int: Tensor length, which is the size of the first dimension.
        """
        return len(self.data)

    def __add__(self, other: Union['Tensor', TensorableT]) -> 'Tensor':
        """Adds two tensors element-wise and returns the result. This tensor is
        the first operand, the other is the second, i.e., 'self + other'.

        Args:
            other (Union['Tensor', TensorableT]): Tensor to be added.

        Returns:
            Tensor: A new tensor containing element-wise sum.
        """
        return add(self, other)

    def __radd__(self, other: Union['Tensor', TensorableT]) -> 'Tensor':
        """Adds two tensors element-wise and returns the result. This tensor is
        the second operand, the other is the first, i.e., 'other + self'.

        Args:
            other (Union['Tensor', TensorableT]): Tensor to be added.

        Returns:
            Tensor: A new tensor containing element-wise sum.
        """
        return add(other, self)

    def __sub__(self, other: Union['Tensor', TensorableT]) -> 'Tensor':
        """Subtracts two tensors in an element-wise fashion and returns the
        result. This tensor is the first operator, the other is the second,
        i.e., 'self - other'.

        Args:
            other (Tensor): Subtrahend.

        Returns:
            Tensor: Element-wise difference of the two tensors.
        """
        return sub(self, other)

    def __rsub__(self, other: Union['Tensor', TensorableT]) -> 'Tensor':
        """Subtracts two tensors in an element-wise fashion and returns the
        result. This tensor is the second operator, the other is the first,
        i.e., 'other - self'.

        Args:
            other (Tensor): Minuend.

        Returns:
            Tensor: Element-wise difference of the two tensors.
        """
        return sub(other, self)

    def __mul__(self, other: Union['Tensor', TensorableT]) -> 'Tensor':
        """Multiplies two tensors element-wise and returns the result. This
        tensor is the first operand, the other is the second,
        i.e., 'self * other'.

        Args:
            other (Union[Tensor, TensorableT]): Tensor to multiply with.

        Returns:
            Tensor: A new tensor containing element-wise multiplication result.
        """
        return mul(self, other)

    def __rmul__(self, other: Union['Tensor', TensorableT]) -> 'Tensor':
        """Multiplies two tensors element-wise and returns the result. This
        tensor is the second operand, the other is the first,
        i.e., 'other * self'.

        Args:
            other (Union[Tensor, TensorableT]): Tensor to multiply with.

        Returns:
            Tensor: A new tensor containing element-wise multiplication result.
        """
        return mul(other, self)

    def __neg__(self) -> 'Tensor':
        """Negates the tensor by computing '-tensor' (swapping a sign) for
        each element and returns the result.

        Returns:
            Tensor: Tensor with opposite signs at each element.
        """
        return neg(self)

    @property
    def data(self) -> np.ndarray:
        """Accesses the underlying tensor data.

        Returns:
            np.ndarray: Accesses tensor data.
        """
        return self._data

    @property
    def shape(self) -> ShapeT:
        """Accesses shape of the underlying tensor structure.

        Returns:
            ShapeT: Tensor shape.
        """
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Accesses the number of dimensions (rank) of the tensor.

        Returns:
            int: The number of dimensions (tensor rank).
        """
        return self._data.ndim

    def zero_grad(self) -> None:
        """Sets gradient to a tensor with a shape identical to the data filled
        with zeroes.
        """
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float32))

    def backward(
        self, grad: Optional[Union['Tensor', TensorableT]] = None
    ) -> None:
        """Computes gradients throughout the entire computational graph starting
        from this node. Gradients are accumulated in the grad member variable.

        Args:
            grad (Optional[Union['Tensor', TensorableT]], optional): Upstream
                gradient. Defaults to None.

        Raises:
            InvalidTensorState: Raised if this tensor has requires_grad set to 
                False yet backward has been called.
            InvalidGradient: Raised if this tensor is not a scalar and the
                upstream gradient has not been specified. 
        """
        if not self.requires_grad:
            raise InvalidTensorState(
                "calling backward on a tensor that does not require gradients"
            )

        if grad is None:
            if self.shape == ():
                # This tensor is a scalar, thus we can compute the gradient with
                # respect to a multiplicative identity.
                grad = Tensor(1.0)
            else:
                # If this tensor is not a scalar, then we need the upstream
                # gradient with multiple elements to compute the current
                # gradient.
                raise InvalidGradient(
                    "a gradient parameter must be specified for a"
                    "non-scalar tensor"
                )

        if self.grad is None:
            # Lazy initialization for the gradients. This could also be done in
            # the constructor, but the gradient with a shape of the data would
            # be held in memory all the time.
            self.zero_grad()

        if not isinstance(grad, Tensor):
            # For simplicity, tensors do not have to be explicitly passed as a
            # parameter.
            grad = Tensor(grad)

        # Accumulate gradients. Unless explicitly cleared, gradients are
        # accumulated. Among other obvious things, it is useful for gradient
        # accumulation, too.
        self.grad._data = self.grad.data + grad.data

        # Backprop starts from the current node and proceeds towards the leaves.
        for dependency in self._iter_dependencies_if_exist():
            # For each dependency, compute the current gradient function with
            # respect to that particular dependency using its provided grad_fn.
            #
            # If we assume that the current node is y, and the previous node
            # tied by a dependency is x, the grad_fn computes the dy/dx.
            backward_grad = Tensor(dependency.grad_fn(grad.data))
            # Then call backward recursively along the computational graph.
            dependency.tensor.backward(backward_grad.data)

    def reduce_sum(self) -> 'Tensor':
        """Takes a tensor and returns the 0-tensor (scalar) representing a sum
        of all its elements.

        Returns:
            Tensor: Sum of all the elements within the tensor.
        """
        return reduce_sum(self)

    def add(self, other: Union['Tensor', TensorableT]) -> 'Tensor':
        """Adds two tensors element-wise and returns the result.

        Args:
            other (Union['Tensor', TensorableT]): Tensor to be added.

        Returns:
            Tensor: A new tensor containing element-wise sum.
        """
        return add(self, other)

    def sub(self, other: Union['Tensor', TensorableT]) -> 'Tensor':
        """Subtracts two tensors in an element-wise fashion and returns the
        result.

        Args:
            other (Tensor): Subtrahend.

        Returns:
            Tensor: Element-wise difference of the two tensors.
        """
        return sub(self, other)

    def mul(self, other: Union['Tensor', TensorableT]) -> 'Tensor':
        """Multiplies two tensors element-wise and returns the result.

        Args:
            other (Union[Tensor, TensorableT]): Tensor to multiply with.

        Returns:
            Tensor: A new tensor containing element-wise multiplication result.
        """
        return mul(self, other)

    def neg(self) -> 'Tensor':
        """Negates the tensor by computing '-tensor' (swapping a sign) for
        each element and returns the result.

        Returns:
            Tensor: Tensor with opposite signs at each element.
        """
        return neg(self)

    def _iter_dependencies_if_exist(self) -> Iterator[Dependency]:
        if self.dependencies:
            yield from iter(self.dependencies)


def reduce_sum(tensor: Tensor) -> Tensor:
    """Takes a tensor and returns the 0-tensor (scalar) representing the sum
    of all its elements.

    Args:
        tensor (Tensor): A tensor to compute the sum of.

    Returns:
        Tensor: A 0-tensor representing the sum.
    """
    ret_data = np.sum(tensor.data)

    # If the tensor depends on anything, then any other tensor produced from
    # this one needs to copy the requires_grad value so as to assure the
    # computational graph remains intact.
    requires_grad = tensor.requires_grad

    if requires_grad:
        dependencies = [Dependency(tensor, _build_reduce_sum_grad_fn(tensor))]
    else:
        dependencies = None

    return Tensor(ret_data, requires_grad, dependencies)


def add(addend_1: Tensor, addend_2: Tensor) -> Tensor:
    """Adds two tensors in an element-wise fashion and returns the result.

    Args:
        addend_1 (Tensor): First addend.
        addend_2 (Tensor): Second addend.

    Returns:
        Tensor: Element-wise sum of the two tensors.
    """
    # This function has to take care of broadcasting.
    # Let X be a 2-by-3 matrix and y be a vector consisting of 3 elements.
    # Let Z = X + y. This operation is valid if broadcasting is used.
    #
    # X = | x1 x2 x3 |    y = | y1 y2 y3 |
    #     | x4 x5 x6 |
    # Z = | (x1 + y1) (x2 + y2) (x3 + y3) |
    #     | (x4 + y1) (x5 + y2) (x6 + y3) |
    #
    # The purpose of broadcasting is to adjust the shapes to equal number of
    # dimensions by adding "ones" where necessary. Then, when "copying" the
    # values, should any of the dimensions be equal to one, then it is expanded
    # to match the other dimension.

    # For example, assume shapes (3, 2, 5) and (5,). The latter would be
    # broadcasted into the former as follows:
    #     1.) Shape adjustment. Right-align dimensions and append "ones" to the
    #         beginning. So (5,) becomes (1, 1, 5).
    #     2.) Expansion. Proceeed from the end, and whenever one of the
    #         dimensions is one, expand to match the other.
    #         So, (1, 1, 5) --> (1, 2, 5) --> (3, 2, 5).
    #
    # This is important when computing gradients. If broadcasting happens, it
    # has to be taken into consideration as the influence of a broadcasted
    # element is magnified.
    ret_data = addend_1.data + addend_2.data

    # If at least one of the tensors requires gradient, then the output needs to
    # propagate this property down the computational graph.
    requires_grad = addend_1.requires_grad or addend_2.requires_grad
    dependencies = [] if requires_grad else None

    if addend_1.requires_grad:
        grad_fn_1 = _build_add_grad_fn(addend_1)
        dependencies.append(Dependency(addend_1, grad_fn_1))

    if addend_2.requires_grad:
        grad_fn_2 = _build_add_grad_fn(addend_2)
        dependencies.append(Dependency(addend_2, grad_fn_2))

    return Tensor(ret_data, requires_grad, dependencies)


def sub(minuend: Tensor, subtrahend: Tensor) -> Tensor:
    """Subtracts two tensors in an element-wise fashion and returns the result.

    Args:
        minuend (Tensor): Minuend.
        subtrahend (Tensor): Subtrahend.

    Returns:
        Tensor: Element-wise difference of the two tensors.
    """
    return add(minuend, neg(subtrahend))


def mul(multiplicand: Tensor, multiplier: Tensor):
    """Multiplies two tensors in an element-wise fashion and returns the result.

    Args:
        multiplicand (Tensor): Multiplicand.
        multiplier (Tensor): Multiplier.

    Returns:
        Tensor: Element-wise product of the two tensors.
    """
    ret_data = multiplicand.data * multiplier.data

    requires_grad = multiplicand.requires_grad or multiplier.requires_grad
    dependencies = [] if requires_grad else None

    if multiplicand.requires_grad:
        grad_fn_1 = _build_mul_grad_fn(multiplicand, multiplier)
        dependencies.append(Dependency(multiplicand, grad_fn_1))

    if multiplier.requires_grad:
        grad_fn_2 = _build_mul_grad_fn(multiplier, multiplicand)
        dependencies.append(Dependency(multiplier, grad_fn_2))

    return Tensor(ret_data, requires_grad, dependencies)


def neg(tensor: Tensor) -> Tensor:
    """Negates the tensor by computing '-tensor' (swapping a sign) for
    each element and returns the result.

    Args:
        tensor (Tensor): Tensor to be negated.

    Returns:
        Tensor: Tensor with opposite signs at each element.
    """
    ret_data = -tensor.data
    requires_grad = tensor.requires_grad

    if requires_grad:

        def _grad_fn(grad: np.ndarray) -> np.ndarray:
            """Computes a gradient with respect to a negated tensor.

            Args:
                grad (np.ndarray): Upstream gradient.

            Returns:
                np.ndarray: Gradient with respect to a negated tensor.
            """
            # y = -x --> dy/dx = -1
            return -grad

        dependencies = [Dependency(tensor, _grad_fn)]
    else:
        dependencies = None

    return Tensor(ret_data, requires_grad, dependencies)


def _build_reduce_sum_grad_fn(tensor: Tensor) -> GradFnT:
    def _grad_fn(grad: np.ndarray) -> np.ndarray:
        """Computes a gradient with respect to the specified tensor as part of
        the reduce_sum operation.

        It accepts an upstream gradient that has to be a scalar, i.e., a
        0-dimensional tensor.

        Args:
            grad (np.ndarray): Upstream gradient (a scalar by definition).

        Returns:
            np.ndarray: Gradient with respect to the specified tensor.
        """
        # Let S be the computed sum, so
        #     S = x1 + x2 + ... + xn.
        # Taking the derivative with respect to any element is equal to one.
        # Thus, each element contributes equally to the sum.
        #     dS/x = [dS/dx1, dS/dx2, ..., dS/dxn] = [1, 1, ..., 1]
        # However, if the upstream gradient is an arbitrary scalar c, then
        # the contribution is equal to
        #    dS/x = c * [1, 1, ..., 1]
        return grad * np.ones_like(tensor.data)

    return _grad_fn


def _build_add_grad_fn(tensor: Tensor) -> GradFnT:
    def _grad_fn(grad: np.ndarray) -> np.ndarray:
        """Computes the gradient with respect to the specified tensor in the
        outer scope as part of the add operation. It handles broadcasting by
        magnifying the contribution of individual elements along broadcasted
        dimensions.

        Args:
            grad (np.ndarray): Upstream gradient.

        Returns:
            np.ndarray: Gradient with respect to the specified tensor.
        """
        # In the case of addition, just simple gradient propagation is needed.
        return _accum_grad_after_broadcast_if_needed(tensor, grad)

    return _grad_fn


def _build_mul_grad_fn(target_tensor: Tensor, other_tensor: Tensor) -> GradFnT:
    def _grad_fn(grad: np.ndarray) -> np.ndarray:
        """Computes the gradient with respect to the target_tensor specified in
        the outer scope as part of the mul operation. It handles broadcasting by
        magnifying the contribution of individual elements along broadcasted
        dimensions.

        Args:
            grad (np.ndarray): Upstream gradient.

        Returns:
            np.ndarray: Gradient with respect to the target_tensor.
        """
        # If z = x * y, then dz/dx = y and dz/dy = x. This is the reason why we
        # specifically consider target and the "other" tensor. Target is the one
        # with respect to which we are computing the gradient.
        grad_curr = grad * other_tensor.data

        return _accum_grad_after_broadcast_if_needed(target_tensor, grad_curr)

    return _grad_fn


def _accum_grad_after_broadcast_if_needed(
    tensor: Tensor, grad: np.ndarray
) -> np.ndarray:
    """Accumulates gradients if broadcasting occurred, which is automatically
    infererred from the shape of tensor and the gradient. The broadcasting
    magnifies element contribution within the computation as it may end up being
    copied to multiple other dimensions. The function is useful for computing
    gradients. Tensor shape has to be broadcastable to gradient shape.

    Args:
        tensor (Tensor): Tensor that was part of the computation with respect to
            which we want to compute the gradient.
        grad (np.ndarray): Upstream gradient of the computation.
    
    Raises:
        InvalidGradient: Raised is the tensor and gradient shapes are not
            broadcastable.

    Returns:
        np.ndarray: Gradient with respect to the specified tensor adjusted by
        the effect of broadcasting if it occurred.
    """
    if not are_broadcastable(tensor.shape, grad.shape):
        raise InvalidGradient(
            "upstream gradient is not broadcastable to the tensor with respect "
            "to which the current gradient is being computed"
        )

    # Compute the number of prepended dimensions.
    n_dims_prepended = grad.ndim - tensor.ndim

    if n_dims_prepended > 0:
        # Sum across the prepended dimensions.
        grad_accum = grad.sum(axis=tuple(range(n_dims_prepended)))
    else:
        grad_accum = grad

    # Broadcasting may happen also within the tensor. For example,
    # assume shapes (3, 2, 5, 4) and (2, 1, 4). The broadcasting first
    # adjusts the number of dimensions, thus (2, 1, 4) --> (1, 2, 1, 4).
    # Then, during expansion, any dimension equal to one is broadcasted,
    # so (1, 2, 1, 4) --> (1, 2, 5, 4) --> (3, 2, 5, 4).
    for axis, dim in enumerate(tensor.shape):
        if dim == 1:
            grad_accum = grad_accum.sum(axis=axis, keepdims=True)

    return grad_accum
