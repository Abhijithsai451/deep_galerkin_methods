import network
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

