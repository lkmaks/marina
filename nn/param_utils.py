import torch
import pickle


def add_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    return z

def sub_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] - y[i])
    return z

def mult_param(alpha, x):
    z = []
    for i in range(len(x)):
        z.append(alpha*x[i])
    return z

def zero_params_like(x):
    z = []
    for i in range(len(x)):
        z.append(torch.zeros(x[i].shape).to(x[i].device))
    return z

def norm_of_param(x):
    z = 0
    for i in range(len(x)):
        z += torch.norm(x[i].flatten(0))
    return z



def copy_module(model):
    return pickle.loads(pickle.dumps(model))
