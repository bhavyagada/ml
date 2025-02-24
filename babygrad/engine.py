from __future__ import annotations
from collections.abc import Callable

class Value:
  def __init__(self, data:int|float, _children:tuple=(), _op:str='') -> None:
    self.data: int|float = data
    self.grad: float = 0
    self._backward: Callable[[], None] = lambda: None
    self._prev: set[Value] = set(_children)
    self._op: str = _op

  def __add__(self, other) -> Value:
    other_val: Value = other if isinstance(other, Value) else Value(other)
    out: Value = Value(self.data + other_val.data, (self, other_val), '+')
    def _backward() -> None:
      self.grad += out.grad
      other_val.grad += out.grad
    out._backward = _backward
    return out

  def __mul__(self, other) -> Value:
    other_val: Value = other if isinstance(other, Value) else Value(other)
    out: Value = Value(self.data * other_val.data, (self, other_val), '*')
    def _backward() -> None:
      self.grad += other_val.data * out.grad
      other_val.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other) -> Value:
    assert isinstance(other, (int, float)), "only supports int/float powers"
    out: Value = Value(self.data**other, (self,), f'**{other}')
    def _backward() -> None:
      self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    return out

  def relu(self) -> Value:
    out: Value = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
    def _backward() -> None:
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out

  def backward(self) -> None:
    topo: list[Value] = []
    visited: set[Value] = set()
    def build_topo(v) -> None:
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad += 1.0
    for v in reversed(topo):
      v._backward()

  def __radd__(self, other: Value|int|float) -> Value: return self + other
  def __rmul__(self, other: Value|int|float) -> Value: return self * other
  def __neg__(self) -> Value: return self * -1 # (-self)
  def __sub__(self, other: Value|int|float) -> Value: return self + (-other)
  def __rsub__(self, other: Value|int|float) -> Value: return other + (-self) # other - self
  def __rpow__(self, other: int|float) -> Value: return Value(other**self.data) # other ** self
  def __truediv__(self, other: Value|int|float) -> Value: return self * other**-1
  def __rtruediv__(self, other: Value|int|float) -> Value: return other * self**-1 # other / self
  def __repr__(self) -> str: return f"Value(data={self.data}, grad={self.grad})"
