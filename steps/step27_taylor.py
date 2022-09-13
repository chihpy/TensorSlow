if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import numpy as np
from tensorslow import Variable, Function
from tensorslow.utils import plot_dot_graph

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

x = Variable(np.array(np.pi/4))
y = sin(x)
taylor_y = my_sin(x)
y.backward()

print(y.data)
print(x.grad)
x.cleargrad()
taylor_y.backward()
print('-taylor result')
print(taylor_y.data)
print(x.grad)
plot_dot_graph(taylor_y, verbose=False, to_file='../figures/taylor_sin_1e-4.png')
taylor_y_1e150 = my_sin(x, 1e-150)
plot_dot_graph(taylor_y_1e150, verbose=False, to_file='../figures/taylor_sin_1e-150.png')