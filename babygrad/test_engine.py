import torch
import unittest
from engine import Value

class TestValueOps(unittest.TestCase):
  a = Value(2.0)
  b = 3.0
  c = Value(-2.0)

  def test_add(self):
    self.assertEqual((self.a + self.b).data, 5.0, "__add__ failed")
    self.assertEqual((self.b + self.a).data, 5.0, "__radd__ failed")

  def test_mul(self):
    self.assertEqual((self.a * self.b).data, 6.0, "__mul__ failed")
    self.assertEqual((self.b * self.a).data, 6.0, "__rmul__ failed")

  def test_sub(self):
    self.assertEqual((self.a - self.b).data, -1.0, "__sub__ failed")
    self.assertEqual((self.b - self.a).data, 1.0, "__rsub__ failed")

  def test_truediv(self):
    self.assertEqual((self.a / self.b).data, 2.0 / 3.0, "__truediv__ failed")
    self.assertEqual((self.b / self.a).data, 3.0 / 2.0, "__rtruediv__ failed")

  def test_pow(self):
    self.assertEqual((self.a ** self.b).data, 8.0, "__pow__ failed")
    self.assertEqual((self.b ** self.a).data, 9.0, "__rpow__ failed")

  def test_relu(self):
    self.assertEqual(self.a.relu().data, 2.0, "ReLU positive failed")
    self.assertEqual(self.c.relu().data, 0.0, "ReLU negative failed")

  def test_exp(self):
    x = Value(2.0)
    y = x.exp()
    y.backward()

    xpt = torch.tensor(x.data, requires_grad=True, dtype=torch.double)
    ypt = torch.exp(xpt)
    ypt.backward()

    self.assertAlmostEqual(y.data, ypt.item(), places=6, msg="exp forward pass failed")
    self.assertAlmostEqual(x.grad, xpt.grad.item(), places=6, msg="exp backward pass failed")

  def test_tanh(self):
    x = Value(2.0)
    y = x.tanh()
    y.backward()

    xpt = torch.tensor(x.data, requires_grad=True, dtype=torch.double)
    ypt = torch.tanh(xpt)
    ypt.backward()

    self.assertAlmostEqual(y.data, ypt.item(), places=6, msg="tanh forward pass failed")
    self.assertAlmostEqual(x.grad, xpt.grad.item(), places=6, msg="tanh backward pass failed")

  def test_sanity_check(self): # from micrograd tests
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    self.assertEqual(ymg.data, ypt.data.item(), "forward pass failed")
    self.assertEqual(xmg.grad, xpt.grad.item(), "backward pass failed")

  def test_more_ops(self): # from micrograd tests
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    self.assertTrue(abs(gmg.data - gpt.data.item()) < tol, "foward pass failed")
    self.assertTrue(abs(amg.grad - apt.grad.item()) < tol, "backward pass failed")
    self.assertTrue(abs(bmg.grad - bpt.grad.item()) < tol, "backward pass failed")

if __name__ == '__main__':
  unittest.main()