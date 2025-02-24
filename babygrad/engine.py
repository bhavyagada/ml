from __future__ import annotations
import math

class Value:
  def __init__(self, data, _children=(), _op='') -> None:
    self.data = float(data)
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op

  def __add__(self, other) -> Value:
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    return out

  def __mul__(self, other) -> Value:
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other) -> Value:
    assert isinstance(other, (int, float)), "only supports int/float powers"
    out = Value(self.data**other, (self,), f'**{other}')
    def _backward():
      self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    return out

  def exp(self) -> Value:
    x = self.data
    out = Value(math.exp(x), (self,), 'exp')
    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out

  def relu(self) -> Value:
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out

  def tanh(self) -> None:
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    return out

  def backward(self) -> None:
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for v in reversed(topo):
      v._backward()

  def __radd__(self, other) -> Value: return self + other
  def __rmul__(self, other) -> Value: return self * other
  def __neg__(self) -> Value: return self * -1 # (-self)
  def __sub__(self, other) -> Value: return self + (-other)
  def __rsub__(self, other) -> Value: return other + (-self) # other - self
  def __rpow__(self, other) -> Value: return Value(other**self.data) # other ** self
  def __truediv__(self, other) -> Value: return self * other**-1
  def __rtruediv__(self, other) -> Value: return other * self**-1 # other / self
  def __repr__(self) -> str: return f"Value(data={self.data}, grad={self.grad})"
