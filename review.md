## Implementation of layers in numpy
```python
import numpy as np

class Sigmoid:
  def forward(self, x):
    clipped = np.clip(x, -500, 500)
    self.output = 1 / (1 + np.exp(-clipped))
    return self.output

  def backward(self, grad_in):
    return self.output * (1 - self.output) * grad_in

#####################################################

class Softmax:
    def forward(self, x):
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_in):
        batch_size = self.output.shape[0]
        grad_out = np.zeros_like(self.output)

        for i in range(batch_size):
            s = self.output[i]
            jacobian = np.diag(s) - np.outer(s, s)
            grad_out[i] = jacobian @ grad_in[i]

        return grad_out
    # More Optimized -- Vectorized
    def backward_vectorized(self, grad_in):
        s = self.output
        dot = np.sum(s * grad_in, axis=1, keepdims=True)
        return s * (grad_in - dot)

#####################################################

class CrossEntropyLoss:
    def forward(self, predictions, targets):
        if targets.ndim == 1:
            self.targets = np.eye(predictions.shape[1])[targets]
        else:
            self.targets = targets

        self.predictions = np.clip(predictions, 1e-12, 1. - 1e-12)
        loss = np.sum(self.targets * np.log(self.predictions), axis=1)
        
        return -np.mean(loss)

    def backward(self):
        batch_size = self.targets.shape[0]
        return  -self.targets / self.predictions / batch_size


####################[Linear Layer]###########################
class Linear:
  def __init__(self, in_features, out_features):
    # Up to you to save self.W in a transposed form or not
    self.W = np.random.randn(in_features, out_features)
    self.b = np.random.randn(out_features)

  def forward(self, x):
    self.x = x
    self.output = x @ self.W
    self.output += self.b
    return self.output

  def backward(self, grad_in):
    self.grad_W = self.x.T @ grad_in
    self.grad_b = np.sum(grad_in, axis=0)
    self.grad_in = grad_in @ self.W.T
    return self.grad_in
```
