import torch
def neural_network(self, X, weights, biases):
    num_layers = len(weights) + 1
    H = 2.0 * (X - self.lb)/(self.ub - self.lb) - 1.0
    for l in range(0, num_layers -2):
        W = weights[l]
        b = biases[l]
        H = torch.tanh(torch.dot(H, W) + b)
    W = weights[-1]
    b = biases[-1]
    Y = torch.add(torch.matmul(H, W), b)
    return Y

