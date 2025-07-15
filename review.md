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


## Selection of Graph Questions
| Status | Problem | Number | Key Algorithm/Pattern | Time Complexity | Notes |
|--------|---------|--------|----------------------|-----------------|-------|
|✔️| Number of Islands | 200 | DFS/BFS | O(V+E) | Grid traversal, mark visited |
|✔️| Number of Provinces | 547 | DFS/Union-Find | O(V²) | Adjacency matrix traversal |
|✔️| Number of Connected Components | 323 | DFS/Union-Find | O(V+E) | Count | components in undirected graph |
|✔️| Graph Valid Tree | 261 | DFS/Union-Find | O(V+E) | Check n-1 edges, no cycles, connected |
|✔️| Redundant Connection | 684 | Union-Find | O(α(n)) | Find edge that creates cycle |
|✔️| Is Graph Bipartite? | 785 | DFS/BFS Coloring | O(V+E) | 2-coloring problem |
|✔️| Possible Bipartition | 886 | DFS/BFS Coloring | O(V+E) | Bipartite graph variant |
|✔️| Course Schedule | 207 | Topological Sort (Kahn/DFS) | O(V+E) | Cycle detection in directed graph |
|✔️| Course Schedule II | 210 | Topological Sort (Kahn/DFS) | O(V+E) | Return topological ordering |
|✔️| Alien Dictionary | 269 | Topological Sort (Kahn) | O(V+E) | Early return if word2 prefix of word1 <br> check if all letters are processed <br>⚠️We don't have to check if an edige already exists. WHY?|
|⚠️✔️| Critical Connections | 1192 | Tarjan's Algorithm | O(V+E) | Find bridges in graph |
|✔️| Find Eventual Safe States | 802 | Topological Sort/DFS | O(V+E) | Reverse graph, find nodes with no outgoing edges. ⚠️ yet better use `DFS` |
|⚠️✔️| Network Delay Time | 743 | Dijkstra | O((V+E)logV) | Single source shortest path |
|✔️| Cheapest Flights K Stops | 787 | Bellman-Ford/BFS | O(K*E) | BFS with relaxing cost, 3D approach<br>💡Bellman-Ford is actually pretty neat |
|✔️| Path With Minimum Effort | 1631 | Dijkstra/Binary Search | O(V*logV) | Skip worse efforts after heap pop |
|✔️| Min Cost to Connect Points | 1584 | Prim/Kruskal MST | O(V²)/O(ElogE) | Optimized Prim O(V²), Kruskal better for sparse |
|✔️| Connecting Cities With Minimum Cost | 1135 | Kruskal MST | O(ElogE) | Standard MST problem |
|✔️| Binary Tree Inorder | 94 | DFS Traversal | O(n) | Left, root, right |
|✔️| Level Order | 102 | BFS | O(n) | Queue-based traversal |
|⚠️✔️| Serialize/Deserialize | 297 | DFS/BFS | O(n) | Tree encoding/decoding |
|✔️| LCA of Binary Tree | 236 | DFS | O(n) | Check if root is answer before recursion |
|✔️| LCA of BST | 235 | DFS | O(h) | Use BST property |
|✔️| Binary Tree Max Path Sum | 124 | DFS | O(n) | Only take positive gains |
|✔️| Diameter of Binary Tree | 543 | DFS | O(n) | Max path through any node |
|🔲| Flower Planting With No Adjacent | 1042 | Graph Coloring | O(V+E) | 4-coloring problem |
|🔲| Remove Max Number of Edges | 1579 | Union-Find | O(α(n)) | Keep minimum edges for connectivity |
|🔲| Minimum Days to Disconnect Island | 1568 | Articulation Points/DFS | O(V+E) | Answer is always 0, 1, or 2; find cut vertices |
|| **Find the City** | **1334** | **Dijkstra/Floyd-Warshall** | **O(V³)** | **All pairs shortest path** 💡Floyd Warshall|
| | **Evaluate Division** | **399** | **DFS/Union-Find** | **O(V+E)** | **Weighted graph with ratios** |
| | **Reconstruct Itinerary** | **332** | **DFS/Eulerian Path** | **O(ElogE)** | **Find Eulerian path, lexicographic order** |
| | **All Paths Source to Destination** | **1059** | **DFS** | **O(V+E)** | **Check all paths lead to target** |
| | **Shortest Path Alternating Colors** | **1129** | **BFS** | **O(V+E)** | **State-based BFS with colors** |
| | **Shortest Path in Binary Matrix** | **1091** | **BFS** | **O(V+E)** | **Grid BFS with obstacles** |
| | **Rotting Oranges** | **994** | **Multi-source BFS** | **O(V+E)** | **BFS from multiple starting points** |
| | **Accounts Merge** | **721** | **Union-Find/DFS** | **O(α(n))** | **Group connected components** |
|⚠️| **Smallest String With Swaps** | **1202** | **Union-Find** | **O(α(n))** | **Sort within connected components** |
|❌| **Word Search** | **79** | **Backtracking** | **O(4^L)** | **DFS with backtracking on grid** |
|❌| **Word Search II** | **212** | **Trie + Backtracking** | **O(4^L)** | **Multiple word search with trie** |

Here are specific LeetCode problems for the missing topics:

## Strongly Connected Components
| Status | Problem | Number | Key Algorithm/Pattern | Time Complexity | Notes |
|--------|---------|--------|----------------------|-----------------|-------|
| | **Strongly Connected Components** | **1192** | **Tarjan's Algorithm** | **O(V+E)** | **Critical Connections - find bridges** |
| | **Maximum Students Taking Exam** | **1349** | **Bipartite Matching/DP** | **O(m*2^n)** | **State compression DP** |

## Articulation Points & Bridges
| Status | Problem | Number | Key Algorithm/Pattern | Time Complexity | Notes |
|--------|---------|--------|----------------------|-----------------|-------|
| | **Critical Connections in Network** | **1192** | **Tarjan's Algorithm** | **O(V+E)** | **Find bridges in undirected graph** |

## Advanced Graph Problems
| Status | Problem | Number | Key Algorithm/Pattern | Time Complexity | Notes |
|--------|---------|--------|----------------------|-----------------|-------|
| | **Satisfiability of Equality Equations** | **990** | **Union-Find** | **O(α(n))** | **Process != constraints after == constraints** |
| | **Most Stones Removed** | **947** | **Union-Find/DFS** | **O(α(n))** | **Count connected components, answer = stones - components** |
| | **Swim in Rising Water** | **778** | **Dijkstra/Binary Search + DFS** | **O(V²logV)** | **Modified shortest path with time constraint** |
| | **Bus Routes** | **815** | **BFS** | **O(V+E)** | **Graph of bus routes, not stops** |
| | **Word Ladder** | **127** | **BFS** | **O(V+E)** | **Shortest transformation sequence** |
| | **Word Ladder II** | **126** | **BFS + DFS** | **O(V+E)** | **All shortest transformation sequences** |
| | **Clone Graph** | **133** | **DFS/BFS** | **O(V+E)** | **Deep copy of graph with HashMap** |

## Tree/Graph Traversal & Special Cases
| Status | Problem | Number | Key Algorithm/Pattern | Time Complexity | Notes |
|--------|---------|--------|----------------------|-----------------|-------|
| | **Vertical Order Traversal** | **987** | **BFS/DFS + Sorting** | **O(nlogn)** | **Coordinate-based traversal** |
| | **Binary Tree Right Side View** | **199** | **BFS/DFS** | **O(n)** | **Level order, take rightmost** |
| | **Boundary of Binary Tree** | **545** | **DFS** | **O(n)** | **Left boundary + leaves + right boundary** |

## Advanced Union-Find
| Status | Problem | Number | Key Algorithm/Pattern | Time Complexity | Notes |
|--------|---------|--------|----------------------|-----------------|-------|
| | **Regions Cut by Slashes** | **959** | **Union-Find** | **O(n²α(n))** | **Split each cell into 4 triangles** |
| | **Bricks Falling When Hit** | **803** | **Union-Find (Reverse)** | **O(mn α(mn))** | **Process in reverse order** |

## Graph Coloring & Matching
| Status | Problem | Number | Key Algorithm/Pattern | Time Complexity | Notes |
|--------|---------|--------|----------------------|-----------------|-------|
| | **Flower Planting With No Adjacent** | **1042** | **Graph Coloring** | **O(V+E)** | **4-coloring problem** |
| | **Coloring A Border** | **1034** | **DFS** | **O(mn)** | **Connected component boundary coloring** |

## Advanced Shortest Path
| Status | Problem | Number | Key Algorithm/Pattern | Time Complexity | Notes |
|--------|---------|--------|----------------------|-----------------|-------|
| | **Shortest Path Visiting All Nodes** | **847** | **BFS + Bitmask** | **O(2^n * n²)** | **TSP variant with BFS** |
| | **Minimum Cost to Make Valid Path** | **1368** | **Dijkstra/0-1 BFS** | **O(mn)** | **Grid with directional costs** |

These additions cover the major missing algorithmic patterns in graph theory that commonly appear in technical interviews.
