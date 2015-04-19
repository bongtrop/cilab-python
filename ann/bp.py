import math
import net

SIGMOID = 0
TANH = 1

class bp:
  def __init__(self, net, learning_rate, momentum):
    self.type = net.getType()
    self.net = net
    self.lr = learning_rate
    self.m = momentum
    self.layer = net.getLayer()
    self.lc = [[[0]*max(self.layer)]*max(self.layer)]*len(self.layer)

  def _dfunc(self, y):
    if self.type==SIGMOID:
      return y * (1.0 - y)
    else:
      return 1.0 - y**2

  def setLearningRate(self,x):
    self.lr = x

  def setMomentum(self, x):
    self.m = x

  def backPropagate(self, input, target):
    if len(target)!=self.layer[-1]:
      print len(target)
      print self.layer[-1]
      raise ValueError('Wrong number of target values')

    self.net.process(input)

    nlayer = len(self.layer)

    delta = []
    for i in range(0, nlayer):
      delta.append([0.0] * self.layer[i])

    for i in range(0,self.layer[nlayer-1]):
      node = self.net.getNode(nlayer-1, i)
      error = target[i] - node
      delta[nlayer-1][i] = self._dfunc(node) * error

    for l in range(nlayer-2, 0, -1):
      for i in range(0, self.layer[l]):
        error = 0.0
        for j in range(0, self.layer[l+1]):
          error = error + delta[l+1][j] * self.net.getWeight(l+1, i, j)

        delta[l][i] = self._dfunc(self.net.getNode(l,i)) * error

    for l in range(nlayer-2, -1, -1):
      for i in range(0, self.layer[l]):
        for j in range(0, self.layer[l+1]):
          change = delta[l+1][j] * self.net.getNode(l, i)
          w = self.net.getWeight(l+1, i, j) + self.lr * change + self.m * self.lc[l+1][i][j]
          self.net.setWeight(l+1, i, j, w)
          self.lc[l+1][i][j] = change

      for i in range(0, self.layer[l+1]):
        b = self.net.getBias(l+1, i) + delta[l+1][i]
        self.net.setBias(l+1, i, b)


    error = 0.0
    for i in range(0, len(target)):
      error = error + 0.5 * (target[i] - self.net.getNode(nlayer-1, i))**2

    return error
