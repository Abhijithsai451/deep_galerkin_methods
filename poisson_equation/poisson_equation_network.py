import torch.nn as nn
import torch

#%% Neural Network architecture used in DGM
class NeuralNetwork(nn.Module):
    def __init__(self, output_dim, input_dim, trans1 = "tanh", trans2 = "tanh"):
        super(NeuralNetwork, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        if trans1 == "tanh":
            self.trans1 = torch.nn.Tanh()
        elif trans1 == "relu":
            self.trans1 = torch.nn.ReLU()
        elif trans1 == "sigmoid":
            self.trans1 = torch.nn.Sigmoid()
        else:
            self.trans1 = torch.nn.Tanh()

        if trans2 == "tanh":
            self.trans2 = torch.nn.Tanh()
        elif trans2 == "relu":
            self.trans2 = torch.nn.ReLU()
        elif trans2 == "sigmoid":
            self.trans2 = torch.nn.Sigmoid()
        else:
            self.trans2 = torch.nn.Tanh()
        self.Uz = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
        self.Ug = nn.Parameter(torch.empty(self.input_dim ,self.output_dim))
        self.Ur = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
        self.Uh = nn.Parameter(torch.empty(self.input_dim, self.output_dim))

        self.Wz = nn.Parameter(torch.empty(self.output_dim, self.output_dim))
        self.Wg = nn.Parameter(torch.empty(self.output_dim, self.output_dim))
        self.Wr = nn.Parameter(torch.empty(self.output_dim, self.output_dim))
        self.Wh = nn.Parameter(torch.empty(self.output_dim, self.output_dim))

        self.bz = nn.Parameter(torch.empty(1, self.output_dim))
        self.bg = nn.Parameter(torch.empty(1, self.output_dim))
        self.br = nn.Parameter(torch.empty(1, self.output_dim))
        self.bh = nn.Parameter(torch.empty(1, self.output_dim))

        nn.init.xavier_uniform_(self.Uz)
        nn.init.xavier_uniform_(self.Ug)
        nn.init.xavier_uniform_(self.Ur)
        nn.init.xavier_uniform_(self.Uh)
        nn.init.xavier_uniform_(self.Wz)
        nn.init.xavier_uniform_(self.Wg)
        nn.init.xavier_uniform_(self.Wr)
        nn.init.xavier_uniform_(self.Wh)

        nn.init.zeros_(self.bz)
        nn.init.zeros_(self.bg)
        nn.init.zeros_(self.br)
        nn.init.zeros_(self.bh)

    def forward(self, S: torch.Tensor, X: torch.tensor) -> torch.Tensor:
        '''Compute output of an LSTMLikeLayer for given inputs S, X.

         Args:
                S: output of previous layer
                X: data input

        Returns: A tensor representing the output of the LSTM-like layer.
        '''

        # compute components of LSTM layer output (note H uses a separate activation function)
        Z = self.trans1(X @ self.Uz + S @ self.Wz + self.bz)
        G = self.trans1(X @ self.Ug + S @ self.Wg + self.bg)
        R = self.trans1(X @ self.Ur + S @ self.Wr + self.br)

        H = self.trans2(X @ self.Uh + (S * R) @ self.Wh + self.bh)

        # compute LSTM layer output
        S_new = (torch.ones_like(G) - G) * H + Z * S

        return S_new

#%% Fully Connected Dense Layer
class DenseLayer(nn.Module):
    def __init__(self, output_dim, input_dim, transformation=None):
        super(DenseLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.W = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
        self.b = nn.Parameter(torch.empty(1, self.output_dim))
        if transformation:
            if transformation == "tanh":
                self.transformation = torch.nn.Tanh()
            elif transformation == "relu":
                self.transformation = torch.nn.ReLU()
            elif transformation == "sigmoid":
                self.transformation = torch.nn.Sigmoid()
        else:
            self.transformation = None

        # Initialize weights and biases
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self,X:torch.Tensor)-> torch.Tensor:
        '''Compute output of a dense layer for a given input X.
        Args:
            X: input to layer
        '''
        S = X @ self.W + self.b
        if self.transformation:
            S = self.transformation(S)
        return S


class DGMNet(nn.Module):
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None):
        super(DGMNet, self).__init__()
        self.initial_layer = DenseLayer(layer_width, input_dim, transformation="tanh")
        self.n_layers = n_layers
        self.LSTMLayerList = nn.ModuleList()
        for _ in range(self.n_layers):
            self.LSTMLayerList.append(NeuralNetwork(layer_width, input_dim))
        self.final_layer = DenseLayer(1, layer_width, transformation=final_trans)
    def forward(self,x):
        '''
            Args:
                x: sampled space inputs

            Run the DGM model and obtain fitted function value at the inputs (x)
        '''
        #X = torch.cat([t,x],1)
        S = self.initial_layer(x)
        # Call Intermediate LSTM Layers
        for layer in self.LSTMLayerList:
            S = layer(S, x)

        # The final layer outputs the solution u(t, x)
        result = self.final_layer(S)
        return result

