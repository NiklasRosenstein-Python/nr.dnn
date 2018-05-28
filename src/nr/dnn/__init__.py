# -*- coding: utf8 -*-
# Copyright (c) 2018 Niklas Rosenstein
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

"""
A very minimalistic neural network framework.
"""

__author__ = 'Niklas Rosenstein <rosensteinniklas@gmail.com>'
__version__ = '1.0.0'

import nr.interface
import numpy as np


class Activation(nr.interface.Interface):
  def map(self, values):
    pass
  def derivative(self, weights, mapped_values):
    pass


class Initializer(nr.interface.Interface):
  def init_weights(self, shape):
    pass


class Sigmoid(nr.interface.Implementation):
  nr.interface.implements(Activation)
  def map(self, values):
    return 1.0 / (1.0 + np.exp(-values))
  def derivative(self, weights, mapped_values):
    return mapped_values * (1.0 - mapped_values)


class RandomInitializer(nr.interface.Implementation):
  nr.interface.implements(Initializer)
  def __init__(self, seed=1):
    self.__seed = seed
  def init_weights(self, shape):
    np.random.seed(self.__seed)
    return np.random.random(shape) * 2 - 1


class Layer(nr.interface.Interface):
  shape = nr.interface.attr(tuple)
  def predict(self, context):
    pass
  def adjust(self, expected):
    pass


class InputLayer(nr.interface.Implementation):
  nr.interface.implements(Layer)

  def __init__(self, variable, cols):
    self.variable = variable
    self.shape = (None, cols)

  def predict(self, context):
    data = context[self.variable]
    if data.shape[1:] != self.shape[1:]:
      raise RuntimeError('variable {!r} expected shape (?, {!r}). got {!r}'
                 .format(self.variable, self.shape[1], data.shape))
    return data

  def adjust(self, expected):
    # We don't propagate backwards for the input layer.
    pass


class HiddenLayer(nr.interface.Implementation):
  nr.interface.implements(Layer)

  def __init__(self, input, nodes, activation=None, initializer=None):
    if activation is None:
      activation = Sigmoid()
    if initializer is None:
      initializer = RandomInitializer()
    self.input = input
    self.shape = (input.shape[1], nodes)
    self.activation = activation
    self.weights = initializer.init_weights(self.shape)
    self.x = None
    self.y = None
    self.delta = None

  def predict(self, context):
    self.x = self.input.predict(context)
    assert self.x.shape[1] == self.shape[0], (self.x.shape, self.shape)
    self.y = self.activation.map(np.dot(self.x, self.weights))
    return self.y

  def adjust(self, expected):
    self.error = expected - self.y
    delta = self.error * self.activation.derivative(self.x, self.y)
    self.weights += self.x.T.dot(delta)
    self.input.adjust(self.x + delta.dot(self.weights.T))
