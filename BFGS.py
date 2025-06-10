from Graph import GraphNet
import argparse
import pandas as pd
import json
import torch
import Equations as Eq
import FeynmanEquations as Fq
import time
import numpy as np
from operators import Operators
from utils import load_defaults_config, add_dict_to_argparser
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

def get_data(expr_name, seed=2024, datatype='Eq', outsize=None):
    np.random.seed(seed)
    try:
        if datatype == 'Eq':
            X, y = Eq.func_dict[expr_name](n=100, outsize=outsize)
        else:
            X, y = Fq.s_func_dict[expr_name](n=100)
    except:
        if datatype == 'Eq':
            X, y = Eq.func_dict[expr_name](num=100, outsize=outsize)
        else:
            X, y = Fq.s_func_dict[expr_name](num=100)
    X = torch.tensor(X, dtype=float)
    y = torch.tensor(y, dtype=float)
    return X, y

def constant_optimize(net, X_train, y_train, verbose=True):
    
    
    params = []
    for v in net.V:
        params += [v.weight]


    loss_func = torch.nn.MSELoss()
    optim = torch.optim.LBFGS(params, lr=0.01) 

    def closure():
        optim.zero_grad() 
        loss = loss_func(net(X_train), y_train)  
        loss.backward()  
        return loss
    
    for t in range(1000):
        loss = optim.step(closure)
        
        if t%100 == 0:
            print(f"Iteration {t+1}, Loss: {loss.item()}")
        if torch.isnan(loss).item():
            return net, loss
        
    return  net, loss.item()

defaults = dict()
defaults.update(load_defaults_config())
parser = argparse.ArgumentParser(description='training args.')
parser = add_dict_to_argparser(parser, defaults)
parser.add_argument('--expr', type=str, default='nguyen1')
parser.add_argument('--dataset', type=str, help='data type', default='Eq')
parser.add_argument('--num', type=int, default=0)
parser.add_argument('--outsize', type=float, default=None)
args = parser.parse_args()
expr_name = args.expr
print("expr_name:", expr_name)

X_train, y_train = get_data(expr_name, seed=2024, datatype=args.dataset, outsize=args.outsize)
X_train = X_train.numpy()  
y_train = y_train.numpy()[:, None]
X_test, y_test = get_data(expr_name, seed=2025, datatype=args.dataset, outsize=args.outsize)
X_test = X_test.numpy()
y_test = y_test.numpy()[:, None]


input_num = X_train.shape[1]
output_num = y_train.shape[1]
print("input_num:", input_num)
print("output_num:", output_num)
print("min inputs:{}, max inputs:{}".format(np.min(X_train), np.max(X_train)))

operators = Operators(args.operator_list, args.var_number)

X_train, y_train = torch.tensor(X_train, dtype=float), torch.tensor(y_train, dtype=float)
X_test, y_test = torch.tensor(X_test, dtype=float), torch.tensor(y_test, dtype=float)

path = './new_results/jsonn{}/{}-{}.json'.format(str(args.node_num), expr, str(i))
net = GraphNet(None, None, operators)
try:
    net.load_graph_from_json(path)
    print("Success!")
    print("start to optimize the weights by lbfgs...")
except:
    print("loading graph error!")
    break

try:
    y_pred = net(X_test)
    origin_test_r2 = r2_score(y_test.numpy(), y_pred.squeeze().detach().numpy())


    net, loss = constant_optimize(net, X_train, y_train)

    y_pred = net(X_train)
    train_r2 = r2_score(y_train.numpy(), y_pred.squeeze().detach().numpy())
    y_pred = net(X_test)
    test_r2 = r2_score(y_test.numpy(), y_pred.squeeze().detach().numpy())
except:
    origin_test_r2 = 0.0
    train_r2 = 0.0
    test_r2 = 0.0
print("origin_test_r2:", origin_test_r2)
print("train_r2:", train_r2)
print(" test_r2:", test_r2)
