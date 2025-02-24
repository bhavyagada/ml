import random
from engine import Value

class Module:
  def parameters(self):
    return []

  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0.0

class Neuron(Module):
  def __init__(self, nin, nonlin=True):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
    self.nonlin = nonlin

  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # w * x + b
    return act.tanh() if self.nonlin else act

  def parameters(self):
    return self.w + [self.b]

  def __repr__(self):
    return f"{'Tanh' if self.nonlin else 'Linear'}({len(self.w)})"

class Layer(Module):
  def __init__(self, nin, nout, nonlin=True):
    self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

  def __repr__(self):
    return f"{len(self.neurons)}x{self.neurons[0]}"

class MLP(Module):
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

  def __repr__(self):
    return f"MLP[{', '.join(str(layer) for layer in self.layers)}]"
