# `nr.dnn`

&ndash; A very minimalistic neural network framework.

## Example

```python
import matplotlib.pyplot as plt
import numpy as np
import nr.dnn

def build_network():
    l0 = nr.dnn.InputLayer('data', 3)
    l1 = nr.dnn.HiddenLayer(l0, 8)
    l2 = nr.dnn.HiddenLayer(l1, 1)
    return l2

def train_network(layer, X, Y, epochs):
    errors = []
    for i in range(epochs):
      y = layer.predict({'data': X})
      layer.adjust(Y)
      errors.append(np.mean(np.abs(Y - y)))
    return y, errors

def main():
  X = np.array([[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1],
                [1,0,0]])
  Y = np.array([[0, 1, 1, 0, 1]]).T
  y, errors = train_network(build_network(), X, Y, 2000)

  print('Mean error:', errors[-1])
  print('Predictions:', np.round(y).T)

  plt.plot(errors)
  plt.show()
```

![](https://i.imgur.com/3RHiJka.png)

## Resources

* [A Neural Network in 11 lines of Python (Part 1)](https://iamtrask.github.io/2015/07/12/basic-python-network/)

---

<p align="center">Copyright &copy; 2018 Niklas Rosenstein</p>
