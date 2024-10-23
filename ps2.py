import numpy as np
import torch
from torch import nn
import torch.optim as optim


################################ BEGIN NUMPY STARTER CODE #################################################
def sigmoid(x):
    #Numerically stable sigmoid function.
    #Taken from: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)


def sample_logistic_distribution(x,a):
    #np.random.seed(1)
    num_samples = len(x)
    y = np.empty(num_samples)
    for i in range(num_samples):
        y[i] = np.random.binomial(1,logistic_positive_prob(x[i],a))
    return y

def create_input_values(dim,num_samples):
    #np.random.seed(100)
    x_inputs = []
    for i in range(num_samples):
        x = 10*np.random.rand(dim)-5
        x_inputs.append(x)
    return x_inputs


def create_dataset():
    x= create_input_values(2,100)
    a=np.array([12,12])
    y=sample_logistic_distribution(x,a)

    return x,y

################################ END NUMPY STARTER CODE ####################################################



################################ BEGIN PYTORCH STARTER CODE ################################################

class TorchLogisticClassifier(nn.Module):

  def __init__(self, num_features):
    super().__init__()
    self.weights = nn.Parameter(torch.zeros(num_features))

  def forward(self, x_vector):
    logit = torch.dot(self.weights, x_vector)
    prob = torch.sigmoid(logit)
    return prob


def loss_fn(y_predicted, y_observed):
    return -1 * (y_observed * torch.log(y_predicted)
                 + (1 - y_observed) * torch.log(1 - y_predicted))

def extract_num_features(dataset):
    first_example = dataset[0]
    # first_example is a pair (x,y), where x is a vector of features and y is 0 or 1
    # note that both x and y are torch tensors
    first_example_x = first_example[0]
    first_example_y = first_example[1]
    num_features = first_example_x.size(0)
    return num_features

def nonbatched_gradient_descent(dataset, num_epochs=10, learning_rate=0.01):
    num_features = extract_num_features(dataset)
    model = TorchLogisticClassifier(num_features)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(num_epochs):
        for d_x, d_y in dataset:
            optimizer.zero_grad()
            prediction = model(d_x)
            loss = loss_fn(prediction, d_y)
            loss.backward()
            optimizer.step()
    return model

def generate_nonbatched_data(num_features=3, num_examples=100):
    x_vectors = [torch.randn(num_features) for _ in range(num_examples)]
    prob_val = 0.5 * torch.ones(1)
    y_vectors = [torch.bernoulli(prob_val) for _ in range(num_examples)]

    dataset = list(zip(x_vectors, y_vectors))

    return dataset

def main():
    nonbatched_dataset = generate_nonbatched_data()
    nonbatched_gradient_descent(nonbatched_dataset)

################################ END PYTORCH STARTER CODE ###################################################


# NOTICE: DO NOT EDIT FUNCTION SIGNATURES 
# PLEASE FILL IN FREE RESPONSE AND CODE IN THE PROVIDED SPACES


# PROBLEM 1
def logistic_positive_prob(x,a):
    pass # YOUR CODE HERE

# PROBLEM 2
def logistic_derivative_per_datapoint(y_i,x_i,a,j):
    pass # YOUR CODE HERE

# PROBLEM 3
def logistic_partial_derivative(y,x,a,j):
    pass # YOUR CODE HERE

# PROBLEM 4
def compute_logistic_gradient(a,y,x):
    pass # YOUR CODE HERE

# PROBLEM 5
def gradient_update(a,lr,gradient):
    pass # YOUR CODE HERE

# PROBLEM 6
def gradient_descent_logistic(initial_a,lr,num_iterations,y,x):
    pass # YOUR CODE HERE

# PROBLEM 7
# Free Response Answer Here: 

# PROBLEM 8
# Free Response Answer Here: 

# PROBLEM 9
def batched_gradient_descent(dataset, num_epochs=10, learning_rate=0.01, batch_size=2):
    pass # YOUR CODE HERE

# PROBLEMS 10-12
def split_into_batches(dataset, batch_size):
    pass

def alt_gradient_descent(dataset, num_epochs=10, learning_rate=0.01, batch_size=2):
    num_features = extract_num_features(dataset)
    model = TorchLogisticClassifier(num_features)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    batches = split_into_batches(dataset, batch_size)
    for i in range(num_epochs):
        # optimizer.zero_grad() # 1
        for batch in batches:
            # optimizer.zero_grad() # 2
            for d_x, d_y in batch:
                # optimizer.zero_grad() # 3
                prediction = model(d_x)
                loss = loss_fn(prediction, d_y)
                loss.backward()
                # optimizer.step() # C
            # optimizer.step() # B
        # optimizer.step() # A
    return model

# PROBLEM 10
$$
\mathbf{w}(t+1) = \mathbf{w}(t) - \eta \cdot \frac{1}{m} \sum_{j=1}^{m} \nabla L(\mathbf{w}(t) \mid B_j)
$$

# PROBLEM 11
$$
\mathbf{w}(t+1) = \mathbf{w}(t) - \eta \cdot \sum_{i=1}^{t} \sum_{j=1}^{m} \nabla L(\mathbf{w}(i) \mid B_j)
$$

# PROBLEM 12
$$
\text{alt\_gradient\_descent\_3B} \quad \text{with arguments:} \quad (dataset, \, num\_epochs, \, learning\_rate = lr, \, batch\_size = k)
$$
