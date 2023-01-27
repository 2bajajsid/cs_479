# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Question 1 and 2: Simple Neural Learning

import numpy as np
import matplotlib.pyplot as plt
from utils import UClasses
import copy
# %load_ext autoreload
# %autoreload 2



# # Preliminaries

# ## Dataset

ds = UClasses(n=1000, binary=False)
ds.plot()

ds.targets()


# ## `Operation` class: Activation/Loss functions

class Operation(object):
    '''
     Operation class
     
     This is the abstract base class that other operations should be based on.
    '''
    def __init__(self):
        return
    
    def __call__(self, x):
        raise NotImplementedError

    def derivative(self):
        raise NotImplementedError


# + [markdown] heading_collapsed=true
# ## Activation functions

# + hidden=true
class Identity(Operation):
    '''
     act = Identity()
     
     Creates an Operation object that represents the identity mapping.
     
     Usage:
      act = Identity()
      act(np.array([[1.2, 5.]]))
     produces the numpy array
      [[1.2, 5.]]
    '''
    def __call__(self, z):
        '''
         y = act(z)
         
         Evaluates the identity function, element-by-element, on z.
         
         Input:
          z  is a numpy array
         Output:
          y  is a numpy array the same size as z
        '''
        self.dims = z.shape
        y = copy.deepcopy(z)
        return y
    
    def derivative(self, s=None):
        '''
         act.derivative(s=None)
         
         Computes the derivative of the identity mapping
         element-by-element.
         Note that the __call__ function must be called before this
         function can be called.
         
         Input:
           s       array the same size as z, which multiplies the
                   derivative
           
         Output:
           dactdz  array the same size as z when __call__ was called,
                   and is s times the derivative
           
         Usage:
           dactdz = act.derivative()
           dactdz = act.derivative(s)
        '''
        # Compute the derivatives
        if s is None:
            return np.ones(self.dims)
        else:
            return s


class Softmax(Operation):
    '''
     act = Softmax()

     Creates an Operation object that represents the softmax
     function. The softmax is applied to the rows of the input.

     Usage:
      act = Softmax()
      act(np.array([[0., 0.5]]))
     produces the numpy array
      [0.37754067 0.62245933]
    '''
    def __call__(self, z):
        v = np.exp(z)
        # Compute denominator (sum along rows)
        denom = np.sum(v, axis=1)
        # Softmax formula, duplicating the denom across rows
        self.y = v/np.tile(denom[:,np.newaxis], [1,np.shape(v)[1]])
        # Store self.y so you can use it in derivative
        return self.y

    def derivative(self, s):
        '''
         dydz = act.derivative(s)

         Computes the derivative of the softmax function.
         Note that the __call__ function must be called before this
         function can be called.

         Input:
           s       array the same size as z, which multiplies the
                   derivative
                   NOTE: s is a *mandatory* argument (not optional)
                   NOTE: s should have only a single non-zero element

         Output:
           dydz    array the same size as z when __call__ was called,
                   and is s Hadamard-times the derivative

         Usage:
           dydz = act.derivative(s)
        '''
        idx = np.nonzero(s)[1]  # Find one-hot categories

        # Create empty copies to populate
        s_gamma = np.zeros_like(s)
        y_gamma = np.zeros_like(self.y)
        kronecker = np.zeros_like(s)

        # Compute dy_k/dz_j 
        for j,gamma in enumerate(idx):
            s_gamma[j,:] = s[j,gamma]
            y_gamma[j,:] = self.y[j,gamma]
            kronecker[j,gamma] = 1.
        dydz = s_gamma*y_gamma*(kronecker-self.y)
        return dydz


# + [markdown] heading_collapsed=true
# ## Loss functions

# + hidden=true
class MSE(Operation):
    '''
     E = MSE()
     
     Creates an object that implements the mean squared error loss.
     
     Usage:
      E = MSE()
      loss = E(y, t)
      
     Example:
      y = np.array([[0.5, 0.1],[-0.4, 0.9], [-0.1, 0.4]])
      t = np.array([[0.6, 0.1],[-0.4, 0.7], [-0.1, 0.6]])
      loss = E(y, t)
     produces the value
      0.015  since it equals
      (0.1^2 + 0.2^2 + 0.2^2)/2 / 3
    '''
    def __call__(self, y, t):
        '''
         E.__call__(y, t)  or   E(y, t)
         
         Computes the mean (average) squared error between the outputs
         y and the targets t.
         
         Inputs:
           y  array with one sample per row
           t  array the same size as y
           
         Output:
           loss  MSE loss (scalar)
        '''
        # MSE formula
        self.n_samples = np.shape(t)[0]
        L = np.sum((y-t)**2)/2./self.n_samples
        self.dL = (y-t) / self.n_samples
        return L

    def derivative(self):
        '''
         E.derivative()
         
         Computes and the derivative of the MSE with respect to y.
         Note that the __call__ function must be called before this
         function can be called.
         
         Output:
           dEdy  array the same size as y when __call__ was called
        '''
        # Compute the gradient of MSE w.r.t. network output
        return self.dL

        
class CategoricalCE(Operation):
    '''
     E = CrossEntropy()

     Creates an object that implements the average cross-entropy loss.

     Usage:
      E = CrossEntropy()
      loss = E(y, t)
    '''
    def __call__(self, y, t):
        '''
         E.__call__(y, t)  or   E(y, t)

         Computes the average categorial cross-entropy between the outputs
         y and the targets t.

         Inputs:
           y  array with one sample per row
           t  array the same size as y

         Output:
           loss  average categorical CE loss (scalar)
        '''
        self.t = t
        self.y = y
        return -np.sum(t * np.log(y)) / len(t)
        
    def derivative(self):
        '''
         E.derivative()

         Computes and the derivative of categorical CE with respect to y.
         Note that the __call__ function must be called before this
         function can be called.

         Output:
           dEdy  array the same size as y when __call__ was called
        '''
        return -self.t/self.y / len(self.t)


# + [markdown] heading_collapsed=true
# # Question 2

# + [markdown] hidden=true
# ## (a) Logistic

# + hidden=true
def logistic_function(x):
  return 1/(1+np.exp(-1 * x))

class Logistic(Operation):
    '''
     act = Logistic()
     
     Creates an Operation object that represents the logistic
     function.
     
     Usage:
      act = Logistic()
      act(np.array([0., 0.5]))
     produces the numpy array
      [0.5 , 0.62245933]
    '''
    def __call__(self, z):
        '''
         y = act(z)
         
         Evaluates the logistic function, element-by-element, on z.
         
         Input:
          z  is a numpy array
          
         Output:
          y  is a numpy array the same size as z
        '''
        #===== YOUR CODE HERE =====
        self.dims = z.shape
        self.z = z
        return 1 / (1 + np.exp(-1 * z))  # replace this line
    
    def derivative(self, s=None):
        '''
         act.derivative(s=None)
         
         Computes the derivative of the logistic function
         element-by-element.
         Note that the __call__ function must be called before this
         function can be called.
         
         Input:
           s       array the same size as z, which multiplies the
                   derivative
                   If s is None (or omitted), an array of 1s will be used.
           
         Output:
           dydz    array the same size as z when __call__ was called,
                   containing the derivative, multiplied by s

         Usage:
           dydz = act.derivative()
           dydz = act.derivative(s)
        '''
        #===== YOUR CODE HERE =====
        if s is None:
            return np.ones(self.dims) 
        else:
            return s * (np.exp(-1 * self.z) / (1 + np.exp(-1 * self.z))**2)

act = Logistic()
#print(act(np.array([0., 0.5])))
#print(act.derivative())
#print(act.derivative(np.array([-0.5, 0, 0.5, 1.5])))

# + [markdown] hidden=true
# ## (b) Cross Entropy

# + hidden=true
class CrossEntropy(Operation):
    '''
     E = CrossEntropy()
     
     Creates an object that implements the average cross-entropy loss.
     
     Usage:
      E = CrossEntropy()
      loss = E(y, t)
    '''
    def __call__(self, y, t):
        '''
         E.__call__(y, t)  or   E(y, t)
         
         Computes the average cross-entropy between the outputs
         y and the targets t.
         
         Inputs:
           y  array with one sample per row
           t  array the same size as y
           
         Output:
           loss  average CE loss (scalar)
        '''
        #===== YOUR CODE HERE =====
        L = 0.   # replace this line

        self.t = t
        self.y = y
        return -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y)) / len(t)

    def derivative(self):
        '''
         E.derivative(s=1)
         
         Computes the derivative of cross-entropy with respect to y.
         Note that the __call__ function must be called before this
         function can be called.
         
         Output:
           dEdy  array the same size as y when __call__ was called
        '''
        #===== YOUR CODE HERE =====
        return -((self.t / self.y) - (1-self.t)/(1-self.y)) / len(self.t)   # replace this line

E = CrossEntropy()
y = np.array([0.21, 0.89, 0.11])
t = np.array([0, 1, 0])
loss = E(y, t)


# + [markdown] heading_collapsed=true
# # `Layer` Classes

# + hidden=true
class Layer(object):
    '''
     Layer is an abstract base class for different
     types of layers.
    '''
    def __init__(self):
        return

    def __call__(self, x):
        raise NotImplementedError


class Population(Layer):
    '''
     lyr = Population(nodes, act=Identity())

     Creates a Population layer object.

     Inputs:
       nodes  the number of nodes in the population
       act    activation function (Operation object)
       
     Usage:
       lyr = Population(3, act=Logistic())
       h = lyr(z)
       print(lyr())   # prints current value of lyr.h
    '''

    def __init__(self, nodes, act=Identity()):
        self.nodes = nodes
        self.z = None
        self.h = None
        self.act = act
        self.params = []

    def __call__(self, x=None):
        if x is not None:
            self.z = x
            self.h = self.act(x)
        return self.h


class Connection(Layer):
    '''
     lyr = Connection(from_nodes=1, to_nodes=1)

     Creates a layer of all-to-all connections.

     Inputs:
       from_nodes  number of nodes in source layer
       to_nodes    number of nodes in receiving layer

     Usage:
       lyr = Connection(from_nodes=3, to_nodes=5)
       z = lyr(h)
       lyr.W    # matrix of connection weights
       lyr.b    # vector of biases
    '''

    def __init__(self, from_nodes=1, to_nodes=1):
        super().__init__()

        self.W = np.random.randn(from_nodes, to_nodes) / np.sqrt(from_nodes)
        self.b = np.zeros(to_nodes)
        self.params = [self.W, self.b]

    def __call__(self, x=None):
        if x is None:
            print('Should not call Connection without arguments.')
            return
        P = len(x)
        if P>1:
            return x@self.W + np.outer(np.ones(P), self.b)
        else:
            return x@self.W + self.b


class DenseLayer(Layer):
    '''
     lyr = DenseLayer(from_nodes=1, to_nodes=1, act=Logistic())

     Creates a DenseLayer object, composed of 2 layer objects:
       L1  a Connection layer of connection weights, and
       L2  a Population layer, consisting of nodes that receives current
           from the Connection layer, and apply the activation function

     Inputs:
       from_nodes  how many nodes are in the layer below
       to_nodes    how many nodes are in the new Population layer
       act         activation function (Operation object)
       
     Usage:
       lyr = DenseLayer(from_nodes=3, to_nodes=5)
       h2 = lyr(h1)
       lyr.L1.W        # connection weights
       lyr.L2()        # activities of layer
       lyr.L2.act      # activation function of layer
    '''

    def __init__(self, from_nodes=1, to_nodes=1, act=Logistic()):
        self.L1 = Connection(from_nodes=from_nodes, to_nodes=to_nodes)
        self.L2 = Population(to_nodes, act=act)

    def __call__(self, x=None):
        if x is None:
            return self.L2.h
        else:
            # Calculate and return the operation of the two layers, L1 and L2
            return self.L2(self.L1(x))


# + [markdown] heading_collapsed=true
# # Question 3: `Network` Class

# + hidden=true
class Network(object):
    '''
     net = Network()

     Creates a Network object.
     
     Usage:
       net = Network()
       net.add_layer(L)
       ... (add more layers)
       y = net(x)
       net.lyr[1]    # reference to Layer object
    '''

    def __init__(self):
        self.lyr = []
        self.loss = None

    def add_layer(self, L):
        '''
         net.add_layer(L)
         
         Adds the layer object L to the network.
         
         Note: It is up to the user to make sure the Layer object
               fits with adjacent layers.
        '''
        self.lyr.append(L)

    def __call__(self, x):
        '''
         y = net(x)
         
         Feedforward pass of the network.
         
         Input:
           x  batch of inputs, one input per row
           
         Output:
           y  corresponding outputs, one per row
        '''
        for l in self.lyr:
            x = l(x)
        return x


    def backprop(self, t, lrate=1.):
        '''
         net.backprop(t, lrate=1.)
         
         Using the error between the state of the output layer and
         the targets, this method does a backprop pass, and updates
         the connection weights and biases.
         
         NOTE: This method assumes that the network is in the
               correct state, following a feedforward pass.
         
         Inputs:
           t      batch of targets, one per row
           lrate  learning rate
        '''
        #========== Question 3, part (a) ==========
        # TODO: Complete the code below.
        # You will have to alter all the lines.
        
        # Set up top gradient
        # since the last layer is a dense layer
        network_output = self.lyr[-1]()
        self.loss(network_output, t)
        dEdh = np.zeros_like(self.lyr[-1]())
        dEdh = self.loss.derivative()

        # Work our way down through the layers
        for i in range(len(self.lyr)-1, 0, -1):

            # References to the layer below, and layer above
            pre = self.lyr[i-1]   # layer below, (i-1)
            post = self.lyr[i]    # layer above, (i)
            # Note that:
            #   post.L1.W contains the connection weights
            #   post.L1.b contains the biases
            #   post.L2.z contains the input currents
            #   post.L2.h contains the upper layer's activities

            # Compute dEdz from dEdh
            dEdz = post.L2.act.derivative(dEdh)

            # Parameter gradients
            dEdW = np.zeros_like(post.L1.W)
            if i == 1:
                dEdW = np.matmul(pre.h.transpose(), dEdz) 
            else: 
                dEdW = np.matmul(pre.L2.h.transpose(), dEdz) 

            dEdb = np.zeros_like(post.L1.b)
            dEdb = np.matmul(np.ones((1, 1000)), dEdz)

            # Project gradient through connection, to layer below
            dEdh = np.matmul(dEdz, post.L1.W.transpose())

            # Update weight parameters
            post.L1.W -= (lrate * dEdW)
            post.L1.b -= (lrate * dEdb[0])
        
        
    def learn(self, ds, lrate=1., epochs=5000):
        '''
         net.Learn(ds, lrate=1., epochs=10)

         Runs error backpropagation on the network, training on
         the data from the Dataset object ds.
         
         Inputs:
           ds       a Dataset object
           lrate    learning rate
           epochs   number of epochs to run
        '''
        #========== Question 3, part (b) ==========
        # TODO: Complete the code below.
        # You will have to edit all these lines.

        #targets = ds.targets()[:, [0]]

        #=== MIGHT I INTEREST YOU IN ADDING SOME CODE HERE? ===
        loss_history = []  # for plotting
        for epoch in range(epochs):
            
            #=== YOU'RE GOING TO WANT TO ADD SOME CODE HERE ===
            self(ds.inputs())
            self.backprop(ds.targets())
            network_output = self.lyr[-1]()
            
            # Give the poor user some feedback so they know something
            # is happening. :)
            if epoch%100==0:
                cost = self.loss(network_output, ds.targets())
                loss_history.append(cost)
                print(f'{epoch}: cost = {cost}')

        return np.array(loss_history)  # don't touch this line
            

# -

# # Your code should work below

# +
net = Network()

# Create layers
input_layer = Population(2)
h1 = DenseLayer(from_nodes=2, to_nodes=30, act=Logistic())
h2 = DenseLayer(from_nodes=30, to_nodes=10, act=Logistic())

# Only use one of these output_layer/loss combinations

# Logistic + CrossEntropy
#output_layer = DenseLayer(from_nodes=10, to_nodes=1, act=Logistic())
#net.loss = CrossEntropy()

# Softmax + Categorical CE
output_layer = DenseLayer(from_nodes=10, to_nodes=2, act=Softmax())
net.loss = CategoricalCE()

# Add layers to the network, from bottom to top
net.add_layer(input_layer)
net.add_layer(h1)
net.add_layer(h2)
net.add_layer(output_layer)

# +
# Train the network
loss_history = net.learn(ds, epochs=5000);

# Plot the progress of the cost
plt.plot(loss_history);
plt.xlabel('Epoch')
plt.ylabel('Cost');
# -

# Sanity check, to see if output match targets
y = net(ds.inputs())
print(f'Outputs:\n{y[:5,:]}')
print(f'Targets:\n{ds.targets()[:5,:]}')

# Cluster plot to make us feel accomplished!
ds.plot(labels=y)


# ## Accuracy of your model

# +
def accuracy(y, t):
    '''
     ac = accuracy(y, t)
     
     Calculates the fraction of correctly classified samples.
     A sample is classified correctly if the largest element
     in y corresponds to where the 1 is in the target.
     
     Inputs:
       y  a batch of outputs, with one sample per row
       t  the corresponding batch of targets
       
     Output:
       ac the fraction of correct classifications (0<=ac<=1)
    '''
    true_class = np.argmax(t, axis=1)       # vector of indices for true class
    estimated_class = np.argmax(y, axis=1)  # vector of indices for estimated class
    errors = sum(true_class==estimated_class)  # add up how many times they match
    acc = errors / len(ds)    # divide by the total number of samples
    return acc

def accuracy_2(y, t):
    '''
     ac = accuracy(y, t)
     
     Calculates the fraction of correctly classified samples.
     A sample is classified correctly if the largest element
     in y corresponds to where the 1 is in the target.
     
     Inputs:
       y  a batch of outputs, with one sample per row
       t  the corresponding batch of targets
       
     Output:
       ac the fraction of correct classifications (0<=ac<=1)
    '''
    targets = ds.targets()[:, [0]]
    y = np.rint(y)

    errors = sum(y==targets)  # add up how many times they match
    acc = errors / len(ds)    # divide by the total number of samples
    return acc


# -

ac = accuracy(net(ds.inputs()), ds.targets())
#ac = accuracy_2(net(ds.inputs()), ds.targets())
print(f"Your model's training accuracy = {ac*100}%")
