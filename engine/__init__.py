import math
import random
import sys
sys.setrecursionlimit(2000)

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.gradient = 0
        
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self.previous = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.gradient += out.gradient
            other.gradient += out.gradient
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.gradient += other.data * out.gradient
            other.gradient += self.data * out.gradient
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.gradient += (other * self.data**(other-1)) * out.gradient
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.gradient += (out.data > 0) * out.gradient
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1)/(math.exp(2 * x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.gradient += (1 - t**2) * out.gradient
        out._backward = _backward

        return out
  
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.gradient += out.data * out.gradient
        out._backward = _backward

        return out

    def log(self):
        x = self.data
        out = Value(math.log(x), (self, ), "log")
        
        def _backward():
            self.gradient += out.gradient / (x * math.log(math.exp(1)))
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.previous:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.gradient = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __lt__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data < other.data
    
    def __gt__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data > other.data

    def __repr__(self):
        return f"Value(data={self.data}, gradient={self.gradient})"


class Neuron:
    def __init__(self, n_input):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_input)]
        self.bias = Value(0)
    
    def __call__(self, x):
        output = sum(
            (wi * xi) for wi, xi in zip(self.w, x)
        ) + self.bias
        output = output.tanh()
        return output
    
    def parameters(self):
        return self.w + [self.bias]


class Layer:
    def __init__(self, n_input, n_output):
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __call__(self, x):
        outputs = [
            neuron(x) for neuron in self.neurons
        ]
        return outputs if len(outputs) > 1 else outputs[0]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, n_input, n_outputs):
        size = [n_input] + n_outputs
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
