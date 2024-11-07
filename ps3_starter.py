import numpy as np
import torch
from torch import nn


######################################## BEGIN STARTER CODE ########################################

def relu(x):
    if x<0:
        return 0
    else:
        return x

def loss(y_predicted, y_observed):
    return (y_predicted - y_observed)**2


def mlp(x,W0,W1,W2):

    r0_0 = x*W0[0]
    r0_1 = x*W0[1]
    r0_2 = x*W0[2]
    r0_3 = x*W0[3]
    r0 = np.array([r0_0,r0_1,r0_2,r0_3])

    h0_0 = relu(r0_0)
    h0_1 = relu(r0_1)
    h0_2 = relu(r0_2)
    h0_3 = relu(r0_3)
    h0 = np.array([h0_0,h0_1,h0_2,h0_3])

    r1_0 = h0_0*W1[0,0] + h0_1*W1[0,1]+ h0_2*W1[0,2] + h0_3*W1[0,3]
    r1_1 = h0_0*W1[1,0] + h0_1*W1[1,1]+ h0_2*W1[1,2] + h0_3*W1[1,3]
    r1 = np.array([r1_0,r1_1])

    h1_0 = relu(r1_0)
    h1_1 = relu(r1_1)
    h1 = np.array([h1_0,h1_1])

    y_predicted = h1_0*W2[0] + h1_1*W2[1]

    variable_dict = {}
    variable_dict['x'] = x
    variable_dict['r0'] = r0
    variable_dict['h0'] = h0
    variable_dict['r1'] = r1
    variable_dict['h1'] = h1
    variable_dict['y_predicted'] = y_predicted

    return variable_dict

### Example arguments ###
# x = 10
# W0 = np.array([1,2,3,4])
# W1 = np.array([[3,4,5,6],[-5,4,3,-2]])
# W2 = np.array([1,-3])

######################################## END STARTER CODE ########################################

# NOTICE: DO NOT EDIT FUNCTION SIGNATURES
# PLEASE FILL IN FREE RESPONSE AND CODE IN THE PROVIDED SPACES

#Problem 1
def d_loss_d_ypredicted(variable_dict,y_observed):
    pass # YOUR CODE HERE

#Problem 2
def d_loss_d_W2(variable_dict,y_observed):
    pass # YOUR CODE HERE

#Problem 3
def d_loss_d_h1(variable_dict,W2,y_observed):
    pass # YOUR CODE HERE

#Problem 4
def relu_derivative(x):
    pass # YOUR CODE HERE

#Problem 5
def d_loss_d_r1(variable_dict,W2,y_observed):
    pass # YOUR CODE HERE

#Problem 6
def d_loss_d_W1(variable_dict,W2,y_observed):
    pass # YOUR CODE HERE

#Problem 7
def d_loss_d_h0(variable_dict,W1,W2,y_observed):
    pass # YOUR CODE HERE

#Problem 8
def d_loss_d_r0(variable_dict,W1,W2,y_observed):
    pass # YOUR CODE HERE

#Problem 9
def d_loss_d_W0(variable_dict,W1,W2,y_observed):
    pass # YOUR CODE HERE

#PROBLEM 10
class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        ##YOUR CODE HERE##

    def forward(self, x):
        pass # YOUR CODE HERE

# PROBLEM 11
def torch_loss(y_predicted,y_observed):
    pass # YOUR CODE HERE

# PROBLEM 12
def torch_compute_gradient(x,y_observed,model):
    pass # YOUR CODE HERE

