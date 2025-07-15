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



## Pytorch / Numpy Operations
| NumPy Operation | Description | NumPy Syntax | PyTorch Equivalent |
|----------------|-------------|--------------|-------------------|
| **Array Creation** | Create arrays from lists/tuples | `np.array([1,2,3])` | `torch.tensor([1,2,3])` |
| **Zeros/Ones** | Create arrays filled with 0s/1s | `np.zeros((3,4))`, `np.ones((3,4))` | `torch.zeros(3,4)`, `torch.ones(3,4)` |
| **Random Arrays** | Create random arrays | `np.random.rand(3,4)` | `torch.rand(3,4)` |
| **Arange/Linspace** | Create sequences | `np.arange(0,10,2)`, `np.linspace(0,1,5)` | `torch.arange(0,10,2)`, `torch.linspace(0,1,5)` |
| **Shape/Reshape** | Get/change array dimensions | `arr.shape`, `arr.reshape(2,3)` | `tensor.shape`, `tensor.reshape(2,3)` |
| **Indexing/Slicing** | Access array elements | `arr[0]`, `arr[1:3]`, `arr[:, 0]` | `tensor[0]`, `tensor[1:3]`, `tensor[:, 0]` |
| **Mathematical Ops** | Element-wise operations | `arr1 + arr2`, `arr * 2`, `np.sqrt(arr)` | `tensor1 + tensor2`, `tensor * 2`, `torch.sqrt(tensor)` |
| **Matrix Multiplication** | Dot product/matrix multiply | `np.dot(a,b)`, `a @ b` | `torch.mm(a,b)`, `a @ b`, `torch.matmul(a,b)` |
| **Transpose** | Transpose arrays | `arr.T`, `np.transpose(arr)` | `tensor.T`, `tensor.transpose(0,1)` |
| **Concatenation** | Join arrays | `np.concatenate([a,b])`, `np.stack([a,b])` | `torch.cat([a,b])`, `torch.stack([a,b])` |
| **Reduction Ops** | Sum, mean, max, etc. | `np.sum(arr)`, `np.mean(arr)`, `np.max(arr)` | `torch.sum(tensor)`, `torch.mean(tensor)`, `torch.max(tensor)` |
| **Broadcasting** | Operations on different shapes | `arr + scalar`, automatic expansion | Same behavior with tensors |
| **Boolean Indexing** | Filter with conditions | `arr[arr > 5]`, `np.where(condition)` | `tensor[tensor > 5]`, `torch.where(condition)` |
| **Data Types** | Specify/convert types | `arr.astype(np.float32)`, `arr.dtype` | `tensor.float()`, `tensor.dtype` |
| **Axis Operations** | Operations along specific axes | `np.sum(arr, axis=0)`, `np.argmax(arr, axis=1)` | `torch.sum(tensor, dim=0)`, `torch.argmax(tensor, dim=1)` |
| **Squeeze/Unsqueeze** | Remove/add dimensions of size 1 | `np.squeeze(arr)`, `np.expand_dims(arr, axis=0)` | `tensor.squeeze()`, `tensor.unsqueeze(0)` |
| **Flatten/Ravel** | Convert to 1D array | `arr.flatten()`, `arr.ravel()` | `tensor.flatten()`, `tensor.view(-1)` |
| **Clipping** | Limit values to range | `np.clip(arr, min_val, max_val)` | `torch.clamp(tensor, min_val, max_val)` |
| **Sorting** | Sort arrays | `np.sort(arr)`, `np.argsort(arr)` | `torch.sort(tensor)`, `torch.argsort(tensor)` |
| **Unique Values** | Find unique elements | `np.unique(arr)` | `torch.unique(tensor)` |
| **Array Comparison** | Element-wise comparisons | `np.equal(a, b)`, `np.allclose(a, b)` | `torch.eq(a, b)`, `torch.allclose(a, b)` |
| **NaN Handling** | Deal with NaN values | `np.isnan(arr)`, `np.nanmean(arr)` | `torch.isnan(tensor)`, `torch.nanmean(tensor)` |
| **Copying** | Create array copies | `arr.copy()` | `tensor.clone()`, `tensor.detach()` |
| **Memory Layout** | Contiguous arrays | `np.ascontiguousarray(arr)` | `tensor.contiguous()` |
| **Splitting** | Split arrays | `np.split(arr, 3)`, `np.hsplit(arr, 2)` | `torch.split(tensor, 3)`, `torch.chunk(tensor, 2)` |
| **Padding** | Add padding to arrays | `np.pad(arr, pad_width)` | `torch.nn.functional.pad(tensor, pad)` |
| **Meshgrid** | Create coordinate grids | `np.meshgrid(x, y)` | `torch.meshgrid(x, y)` |
| Linear Algebra - Norm | Vector/matrix norms | `np.linalg.norm(arr)` | `torch.norm(tensor)` |
| Linear Algebra - Inverse | Matrix inverse | `np.linalg.inv(arr)` | `torch.inverse(tensor)` |
| Linear Algebra - Eigenvalues | Eigenvalues/eigenvectors | `np.linalg.eig(arr)` | `torch.eig(tensor)` |
| Advanced Indexing - Take | Take elements by indices | `np.take(arr, indices)` | `torch.take(tensor, indices)` |
| Advanced Indexing - Put | Put values at indices | `np.put(arr, indices, values)` | `tensor.scatter_(dim, indices, values)` |
| Advanced Indexing - Choose | Choose elements from multiple arrays | `np.choose(indices, arrays)` | `torch.gather(tensor, dim, indices)` |
| Statistics - Std Dev | Standard deviation | `np.std(arr)` | `torch.std(tensor)` |
| Statistics - Variance | Variance | `np.var(arr)` | `torch.var(tensor)` |
| Statistics - Percentiles | Percentiles/quantiles | `np.percentile(arr, q)` | `torch.quantile(tensor, q)` |
| Statistics - Histogram | Histograms | `np.histogram(arr)` | `torch.histc(tensor)` |
| Einstein Summation | Powerful tensor operations | `np.einsum('ij,jk->ik', a, b)` | `torch.einsum('ij,jk->ik', a, b)` |
| Roll Elements | Roll array elements | `np.roll(arr, shift)` | `torch.roll(tensor, shift)` |
| Flip Arrays | Reverse arrays | `np.flip(arr)` | `torch.flip(tensor, dims)` |
| Numerical Gradient | Numerical gradients | `np.gradient(arr)` | No direct equivalent (use autograd) |
| 1D Convolution | 1D convolution | `np.convolve(a, b)` | `torch.nn.functional.conv1d()` |
| FFT | Fast Fourier Transform | `np.fft.fft(arr)` | `torch.fft.fft(tensor)` |
| Polynomial Operations | Polynomial operations | `np.polynomial.polynomial.polyval()` | No direct equivalent |
| Save/Load Arrays | Save/load arrays | `np.savez('file.npz', arr)` | `torch.save(tensor, 'file.pt')` |


## Basic Squeeze/Excitation implementation:
Here's how Squeeze-and-Excitation (SE) is implemented:

## Core SE Block Implementation:

```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        b, c, h, w = x.size()
        
        # Squeeze: Global average pooling
        y = self.squeeze(x)  # (batch, channels, 1, 1)
        y = y.view(b, c)     # (batch, channels)
        
        # Excitation: FC layers
        y = self.excitation(y)  # (batch, channels)
        y = y.view(b, c, 1, 1)  # (batch, channels, 1, 1)
        
        # Scale: Element-wise multiplication
        return x * y.expand_as(x)
```

## Integration with ResNet Block:

```python
class SEResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE block added here
        self.se = SEBlock(out_channels, reduction)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE attention
        out = self.se(out)
        
        out += self.shortcut(x)
        return torch.relu(out)
```

## Key Implementation Details:

- **Reduction ratio**: Typically 16 (reduces channels by 16x in bottleneck)
- **Placement**: Usually after the last conv layer in a block, before residual addition
- **Global pooling**: `AdaptiveAvgPool2d(1)` handles any input size
- **Sigmoid activation**: Ensures attention weights are between 0-1

The SE block is simple but effective - it learns to emphasize important channels and suppress less useful ones with minimal computational overhead.
